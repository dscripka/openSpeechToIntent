# Copyright 2024 David Scripka. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
##################
# Several functions and methods in the files were adapted in whole in or part from several other libraries
# including forced alignment related functions from torchaudio (https://pytorch.org/audio/main/tutorials/ctc_forced_alignment_api_tutorial.html)
# and the excellent pure cpp implementation of the torch forced alignment code by
# @MahmoudAshraf97 (https://github.com/MahmoudAshraf97/ctc-forced-aligner/blob/main/ctc_forced_aligner/forced_align_impl.cpp)

import os
import numpy as np
from typing import List, NamedTuple, Tuple
import json
import pickle
from typing import List, Union
import onnxruntime as ort
import wave
import difflib
from openspeechtointent.forced_alignment import forced_align_single_sequence, forced_align_multiple_sequence


class TokenSpan(NamedTuple):
    """
    A basic class to represent a token span with a score.
    """
    token: int
    start: int
    end: int
    score: float

class CitrinetModel:
    def __init__(self,
                 model_path: str=os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources/models/stt_en_citrinet_256.onnx"),
                 ncpu: int=1
                 ):
        """
        Initialize the Citrinet model, including pre-processing functions.
        Model is obtained from https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/stt_en_citrinet_256
        and then converted to the ONNX format using the standard Nvidia NeMo tools.

        Args:
            model_path (str): Path to the Citrinet model
            ncpu (int): Number of threads to use for inference of the Citrinet model
        """
        # limit to specified number of threads
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = ncpu
        sess_options.inter_op_num_threads = ncpu

        # Load ASR model
        self.asr_model = ort.InferenceSession(model_path, sess_options=sess_options)

        # Load stft model
        location = os.path.dirname(os.path.abspath(__file__))
        self.stft = ort.InferenceSession(os.path.join(location, "resources/models/torchlibrosa_stft.onnx"), sess_options=sess_options)

        # Load filterbank
        filterbank_path = os.path.join(location, "resources/models/citrinet_spectrogram_filterbank.npy")
        self.filterbank = np.load(filterbank_path)

        # Load tokenizer and vocab
        tokenizer_path = os.path.join(location, "resources/models/tokenizer.pkl")
        self.tokenizer = pickle.load(open(tokenizer_path, "rb"))
        vocab_path = os.path.join(location, "resources/models/citrinet_vocab.json")
        self.vocab = json.load(open(vocab_path, 'r'))

        # Initialize similarity matrix attribute
        self.similarity_matrix = None

    def build_intent_similarity_matrix(self, intents: List[str]) -> np.ndarray:
        """Builds a similarity matrix between intents using the longest common subsequence algorithm.

        Args:
            intents (List[str]): List of intents

        Returns:
            np.ndarray: Similarity matrix
        """
        n = len(intents)
        matrix = np.ones((n, n), dtype=float)
        
        for i in range(n):
            for j in range(i+1, n):
                # Calculate similarty using longest common subsequence
                similarity = difflib.SequenceMatcher(None, intents[i], intents[j]).find_longest_match(
                    0, len(intents[i]), 0, len(intents[j])
                ).size/len(intents[i])

                # Fill the matrix symmetrically
                matrix[i][j] = similarity
                matrix[j][i] = similarity
        
        return matrix

    def rerank_intents(self,
                       logits: np.ndarray,
                       intents: List[str],
                       scores: List[float],
                       method: str="longer_match",
                       partial_match_penalty: float=0.1,
                       partial_match_threshold: float=0.1,
                    ):
        """Rerank intents using various hueristics, which can improve accuracy in some cases.

        Args:
            logits (np.ndarray): Logits from the ASR model
            intents (List[str]): List of intents
            scores (List[float]): List of scores for the intents
            method (str): Method to use for reranking. Options are "longer_match" and "partial_match".
                          "partial_match" will rerank intents by penalizing intents that are fully contained within other intents.
                          "longer_match" will rerank intents by preferring longer intents over shorter ones when they have similar scores.
            partial_intent_penalty (float): Score penalty for intents that are fully contained within other intents when using the 
                                            "partial_match" method.

        Returns:
            tuple: Reranked intents and scores (if applicable)
        """

        if method == "longer_match":
            # Reranking intents that are similar in score by the length of the intent
            # This prefers longer matches over shorter ones when there are several very similar options

            buckets = []
            for i, j in zip(intents, scores):
                if buckets == [] or abs(j - buckets[-1][0][1]) >= 0.10:
                    buckets.append([(i, j)])
                else:
                    buckets[-1].append((i, j))

            # Sort buckets by length of intents
            reranked_buckets = [sorted(i, key=lambda x: len(x[0]), reverse=True) if len(i) > 1 else i for i in buckets]
            reranked_intents = [j[0] for i in reranked_buckets for j in i]

            return reranked_intents, []

        if method == "partial_match":
            # See if any intents are completely contained within other longer intents, and if so prefer longer intents
            # by penalizing the score of the shorter intents, but only if the score of the unique portion in the
            # longer intents is above a threshold (that is, is likely also present in the logits)
            new_scores = [i for i in scores]
            for ndx, intent in enumerate(intents):
                if any([intent in j for j in intents if intent != j]):
                    # Get the unique sequence from the longer intents
                    unique_sequences = [j.replace(intent, "").strip() for j in intents if intent != j and intent in j]
                    unique_scores = self.get_forced_alignment_score(logits, unique_sequences + [intent], softmax_scores=True)[1]
                    contained_intent_score = unique_scores[-1]
                    if any([abs(i - contained_intent_score) < 0.1*contained_intent_score for i in unique_scores]):
                        # Penalize the score of the content contained within the other intents
                        new_scores[ndx] -= partial_match_penalty

            # Reorder the intents by the updated scores
            reranked_intents = [intents[i] for i in np.argsort(new_scores)[::-1]]
            reranked_scores = np.sort(new_scores)[::-1].tolist()

            return reranked_intents, reranked_scores

    def match_intents_by_similarity(self,
                        logits: np.ndarray,
                        s: np.ndarray,
                        intents: List[str],
                        sim_threshold: float = 0.6,
                        topk: int = 5,
                        **kwargs
                    ):
        """
        Searches the similarity matrix for intents that are similar,
        and have a score above the threshold. Can reduce the number of calls to the forced alignment models by 30-50% in most cases,
        which reduces total latency.

        Args:
            logits (np.ndarray): Logits from the ASR model used for forced alignment
            s (np.ndarray): Similarity matrix for the intents
            intents (List[str]): List of intents to search
            sim_threshold (float): Similarity threshold to group intents. Lower values will group more intents, which increases efficiency
                                   at the cost of recall. Scores approaching 1 are essentially the same as exhaustive search.
            topk (int): Number of top intents to return
            kwargs: Additional keyword arguments to pass to the `get_forced_alignment_score` function

        Returns:
            tuple: intents, scores, durations (respectively) that meet the thresholds
        """
        # Sort the rows by sum of similarities
        sums = np.sum(s, axis=1)
        sorted_row_indices = np.argsort(sums)

        # Get the score of the intents
        top_intents = []
        top_scores = []
        top_durations = []
        excluded_indices = []
        for ndx in sorted_row_indices:
            if ndx in excluded_indices:
                continue

            # Get score of the intent
            _, score, duration = self.get_forced_alignment_score(logits, [intents[ndx]], softmax_scores=False)

            top_intents.append(intents[ndx])
            top_scores.append(score[0])
            top_durations.append(duration[0])

            # Exclude indices by similarity
            intent_ndcs = np.where(s[ndx, :] > sim_threshold)[0]
            excluded_indices.extend(intent_ndcs)

        # Get the topk results
        topk_score_ndcs = np.array(top_scores).argsort()[::-1][0:topk]
        topk_intents = np.array(top_intents)[topk_score_ndcs]
        topk_scores = np.array(top_scores)[topk_score_ndcs]
        topk_durations = np.array(top_durations)[topk_score_ndcs]

        # Get topk scores and apply softmax to scores
        if kwargs.get("softmax_scores") is True:
            topk_scores = np.round(np.exp(topk_scores)/np.sum(np.exp(topk_scores)), 4)

        return topk_intents, topk_scores, topk_durations

    def get_seq_len(self, seq_len: np.ndarray) -> np.ndarray:
        """
        Get the sequence length for the given input length.
        Note! This has hard-coded values for the default Citrinet 256 model from Nvidia.

        Args:
            seq_len (np.ndarray): Input sequence length

        Returns:
            np.ndarray: Sequence length for the model
        """
        pad_amount = 512 // 2 * 2
        seq_len = np.floor_divide((seq_len + pad_amount - 512), 160) + 1
        return seq_len.astype(np.int64)

    def normalize_batch(self, x: np.ndarray, seq_len: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Normalize the input batch of features.

        Args:
            x (np.ndarray): Input features
            seq_len (np.ndarray): Sequence length
            normalize_type (str): Type of normalization to apply. Options are "per_feature" or "per_batch"

        Returns:
            tuple: Normalized features, mean, and standard deviation
        """
        x_mean = None
        x_std = None
        batch_size, num_features, max_time = x.shape

        time_steps = np.tile(np.arange(max_time)[np.newaxis, :], (batch_size, 1))
        valid_mask = time_steps < seq_len[:, np.newaxis]
        
        x_mean_numerator = np.where(valid_mask[:, np.newaxis, :], x, 0.0).sum(axis=2)
        x_mean_denominator = valid_mask.sum(axis=1)
        x_mean = x_mean_numerator / x_mean_denominator[:, np.newaxis]

        # Subtract 1 in the denominator to correct for the bias.
        x_std = np.sqrt(
            np.sum(np.where(valid_mask[:, np.newaxis, :], x - x_mean[:, :, np.newaxis], 0.0) ** 2, axis=2)
            / (x_mean_denominator[:, np.newaxis] - 1.0)
        )
        # make sure x_std is not zero
        x_std += 1e-5
        return (x - x_mean[:, :, np.newaxis]) / x_std[:, :, np.newaxis], x_mean, x_std

    def get_features(self, x: np.ndarray, length: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the melspectrogram audio features for the raw input audio.

        Args:
            x (np.ndarray): Input audio
            length (np.ndarray): Length of the audio

        Returns:
            tuple: Features and sequence length (both as np.ndarrays)
        
        """
        # get sequence length
        seq_len = self.get_seq_len(length)

        # do preemphasis
        preemph = 0.97
        x = np.concatenate((x[:, 0:1], x[:, 1:] - preemph * x[:, :-1]), axis=1)

        # do stft
        x = np.vstack(self.stft.run([self.stft.get_outputs()[0].name, self.stft.get_outputs()[1].name], {self.stft.get_inputs()[0].name: x}))

        # convert to magnitude
        guard = 0
        x = np.sqrt((x**2).sum(axis=0) + guard).T.squeeze()

        # get power spectrum
        x = x**2

        # dot with filterbank energies
        x = np.matmul(self.filterbank, x)
        
        # log features if required
        x = np.log(x + 5.960464477539063e-08)

        # normalize if required
        x, _, _ = self.normalize_batch(x, seq_len)

        # mask to zero any values beyond seq_len in batch, pad to multiple of `pad_to` (for efficiency)
        max_len = x.shape[-1]
        mask = np.arange(max_len).reshape(1, -1) >= seq_len.reshape(-1, 1)
        x = np.where(mask[:, np.newaxis, :], 0, x)
        
        pad_to = 16
        pad_amt = x.shape[-1] % pad_to
        if pad_amt != 0:
            x = np.pad(x, ((0, 0), (0, 0), (0, pad_to - pad_amt)), mode='constant', constant_values=0)

        return x, seq_len

    def merge_tokens(self, tokens: np.ndarray, scores: np.ndarray, blank: int = 0) -> List[TokenSpan]:
        """Removes repeated tokens and blank tokens from the given CTC token sequence.

        Args:
            tokens (np.ndarray): Alignment tokens (unbatched) returned from forced_align.
                Shape: (time,).
            scores (np.ndarray): Alignment scores (unbatched) returned from forced_align.
                Shape: (time,). When computing the token-size score, the given score is averaged
                across the corresponding time span.

        Returns:
            list of TokenSpan objects
        """
        if tokens.ndim != 1 or scores.ndim != 1:
            raise ValueError("`tokens` and `scores` must be 1D numpy arrays.")
        if len(tokens) != len(scores):
            raise ValueError("`tokens` and `scores` must be the same length.")

        diff = np.diff(np.concatenate(([-1], tokens, [-1])))
        changes_wo_blank = np.nonzero(diff != 0)[0]
        tokens = tokens.tolist()
        spans = [
            TokenSpan(token=token, start=start, end=end, score=np.mean(scores[start:end]))
            for start, end in zip(changes_wo_blank[:-1], changes_wo_blank[1:])
            if (token := tokens[start]) != blank
        ]
        return spans
    
    def get_forced_alignment_score(self,
                                   logits: np.ndarray,
                                   texts: List[str],
                                   topk: int = 5,
                                   softmax_scores: bool = True,
                                   sr: int = 16000,
                                   ncpu: int = 1,
                                ) -> Tuple[List[float], List[float]]:
        """
        Get the forced alignment score for the given logits and text. Scores are optionally softmaxed to so that the 
        score across the topk texts sum to 1.

        Args:
            logits (np.ndarray): Logits from the ASR model
            texts (List[str]): List of texts to align
            topk (int): Number of texts highest score intents to return
            softmax_scores (bool): If True, will apply softmax to the scores
            sr (int): Sample rate of the audio

        Returns:
            tuple: List of text, scores, and durations for best alignment of each text to the logits
        """
        # Get tokens for texts
        new_ids = self.tokenizer.encode(texts)

        # filter out sequences with no tokens
        texts = [text for text, i in zip(texts, new_ids) if i != []]
        new_ids = [i for i in new_ids if i != []]

        # Ensure that tokens are not longer than the time steps in the logits, otherwise truncate
        new_ids = [i if len(i) < logits.shape[0] else i[:logits.shape[0]-1] for i in new_ids]

        # Convert token sequences to numpy arrays with the right shape for forced alignment
        new_ids = [np.array(i)[None,] for i in new_ids]

        # Get forced alignments for all sequences
        alignments = forced_align_multiple_sequence(
            logits[None,],
            new_ids,
            len(self.vocab)-1
        )

        # Get forced alignments
        scores, durations = [], []
        for alignment in alignments:
            # Get token labels for the sequence
            t_labels = alignment[0].flatten()

            # Get the average score of the unmerged sequence of tokens (empirically works better than mean after merging)
            score = round(alignment[1].mean(), 3) 

            # Get the duration of the aligned tokens (don't merge CTC labels as this is fairly slow, and we only need the total duration)       
            non_space_tokens = [ndx for ndx, i in enumerate(t_labels) if i != 1024]
            start = non_space_tokens[0]*1280/sr
            end = (non_space_tokens[-1] + 1)*1280/sr
            duration = round(end - start, 3)

            durations.append(duration)
            scores.append(score)

        # Get topk texts
        sorted_scores_ndcs = np.array(scores).argsort()[::-1][0:topk]
        topk_texts = np.array(texts)[sorted_scores_ndcs]
        topk_scores = np.array(scores)[sorted_scores_ndcs]
        durations = np.array(durations)[sorted_scores_ndcs]

        # Get topk scores and apply softmax to scores
        if softmax_scores is True:
            topk_scores_sm = np.round(np.exp(topk_scores)/np.sum(np.exp(topk_scores)), 4)
            return topk_texts, topk_scores_sm, durations

        return topk_texts, topk_scores, durations

    def get_audio_features(self, audio: Union[str, np.ndarray], sr: int = 16000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the audio features for the given audio file or numpy array.

        Args:
            audio (Union[str, np.ndarray]): Audio file or numpy array of audio
            sr (int): Sample rate of the audio

        Returns:
            tuple: Features and sequence length (both as np.ndarrays)
        """
        if isinstance(audio, str):
            with wave.open(audio, 'rb') as wav_file:
                sr = wav_file.getframerate()
                n_frames = wav_file.getnframes()
                wav_dat = np.frombuffer(wav_file.readframes(n_frames), dtype=np.int16)
        else:
            wav_dat = audio

        # Convert to float32 from 16-bit PCM
        wav_dat = (wav_dat.astype(np.float32) / 32767)
        wav_dat = np.pad(wav_dat, (4300, 4300), mode='constant')  # forced alignment scores seems sensitive to this pad value? Sometimes get seg faults if it is too small?
        all_features, lengths = self.get_features(wav_dat[None,], np.array([wav_dat.shape[0]]))

        return all_features, lengths

    def get_logits(self, audio: Union[str, np.ndarray]) -> np.ndarray:
        """
        Get the logits for the given audio file or numpy array using the Citrinet model.

        Args:
            audio (Union[str, np.ndarray]): Audio file or numpy array of audio

        Returns:
            np.ndarray: Logits from the ASR model
        """
        # Preprocess audio
        all_features, lengths = self.get_audio_features(audio)

        # Transcribe processed audio with the onnx model
        logits = self.asr_model.run(None, {self.asr_model.get_inputs()[0].name: all_features.astype(np.float32), "length": lengths})
        logits = logits[0][0]

        return logits

    def match_intents(self,
                      audio: Union[str, np.ndarray],
                      intents: List[str] = [],
                      topk: int = 5,
                      approximate: bool = False,
                      softmax_scores: bool = True,
                      ) -> Tuple[List[str], List[float], List[float]]:
        """
        Match the intents for the given audio file or numpy array.

        Args:
            audio (Union[str, np.ndarray]): Audio file or numpy array of audio
            intents (List[str]): List of intents to search
            topk (int): Top k intents to return, by score
            approximate (bool): If True, will use approximate intent similarities to more efficiently search for matching intents
            softmax_scores (bool): If True, will apply softmax to the scores. This will make the scores in the topk intents sum to 1.
                                   If false, the scores will be the raw logits values for the forced aligned sequence.

        Returns:
            tuple: List of intents, scores, and durations
        """
        # Get the logits
        logits = self.get_logits(audio)

        # Get the best matching intents
        if approximate is True and intents != []:
            # Build intent similarity matrix and cache for reuse
            if self.similarity_matrix is None:
                self.similarity_matrix = self.build_intent_similarity_matrix(intents)

            top_intents, scores, durations = self.match_intents_by_similarity(
                logits,
                self.similarity_matrix, intents, topk=topk, softmax_scores=softmax_scores
            )

        elif approximate is False and intents != []:
            top_intents, scores, durations = self.get_forced_alignment_score(logits, intents, topk=topk, softmax_scores=softmax_scores)

        return top_intents, scores, durations