# TODO: Add license and disclosures of the source of the original code (mostly torchaudio)

import os
import numpy as np
from typing import List, NamedTuple
import json
import pickle
from typing import List, Union
import onnxruntime as ort
import wave
import difflib
from openspeechtointent.forced_alignment import forced_align


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
                 intents_path: str="",
                 model_path: str=os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources/models/stt_en_citrinet_256.onnx"),
                 ncpu: int=1
                 ):
        """
        Initialize the Citrinet model, including pre-processing functions.
        Model is obtained from https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/stt_en_citrinet_256
        and then converted to the ONNX format using the standard Nvidia NeMo tools.

        Args:
            intents_path (str): Path to the intents JSON file (optional)
            model_path (str): Path to the Citrinet model
            ncpu (int): Number of threads to use for inference
        """
        # limit to single thread
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

        # Load intents assumings a JSON file with a single layer of keys (intent categories), each containing a list of strings
        if intents_path != "":
            if not os.path.exists(intents_path):
                raise FileNotFoundError(f"Intents file not found at {intents_path}")
            self.intents = json.load(open(intents_path, 'r'))

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
                       threshold: float=-5,
                       method: str="word_score"
                    ):
        """Rerank intents using various hueristics, which can improve accuracy in some cases.

        Args:
            logits (np.ndarray): Logits from the ASR model
            intents (List[str]): List of intents
            scores (List[float]): List of scores for the intents
            score_threshold (float): Score threshold for reranking
            method (str): Method to use for reranking. Options are "word_score" and "duration"

        Returns:
            tuple: Reranked intents and scores
        """
        if method == "word_score":
            # Gets all of the words present in the intents and reranks based on the intent with the 
            # higest overlap in words with forced alignment scores above a threshold
            words = set(" ".join(intents).split())
            word_scores = self.get_forced_alignment_score(logits, list(words))[0]
            words = [w for i, w in enumerate(words) if word_scores[i] > threshold]

            word_overlap = []
            for i in range(len(intents)):
                if scores[i] > threshold:
                    word_overlap.append(len(set(intents[i].split()).intersection(words)))
            
            reranked_intents = [intents[i] for i in np.argsort(word_overlap)]
            reranked_scores = [scores[i] for i in np.argsort(word_overlap)]

            return reranked_intents, reranked_scores

    def match_intents_by_similarity(self,
                        logits: np.ndarray,
                        s: np.ndarray,
                        intents: List[str],
                        sim_threshold: float = 0.5,
                        score_threshold: float = -5
                    ):
        """
        Searches the similarity matrix for intents that are similar (by the similarity matrix),
        and have a score above the threshold. Can reduce the number of calls to the forced alignment models by 30-50% in most cases,
        which reduces total latency.

        Args:
            logits (np.ndarray): Logits from the ASR model used for forced alignment
            s (np.ndarray): Similarity matrix for the intents
            intents (List[str]): List of intents to search
            sim_threshold (float): Similarity threshold to group intents. Lower values will group more intents, which increases efficiency
                                   at the cost of recall. Scores approaching 1 are essentially the same as exhaustive search.
            score_threshold (float): Score threshold for the forced alignment. Matches with scores below this threshold will be excluded.

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
            # print(ndx, 107 in excluded_indices)
            if ndx in excluded_indices:
                continue

            # Get score of the intent
            score, duration = self.get_forced_alignment_score(logits, [intents[ndx]])

            # If score is above threshold, add to top intents
            if score[0] >= score_threshold:
                top_intents.append(intents[ndx])
                top_scores.append(score[0])
                top_durations.append(duration[0])

            # Exclude indicies by similarity
            intent_ndcs = np.where(s[ndx, :] > sim_threshold)[0]
            excluded_indices.extend(intent_ndcs)

        # print(n_calls, len(intents))
        return top_intents, top_scores, top_durations

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

    def normalize_batch(self, x: np.ndarray, seq_len: np.ndarray) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
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

        return x, x_mean, x_std

    def get_features(self, x: np.ndarray, length: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
                                   sr: int = 16000,
                                ) -> tuple[List[float], List[float]]:
        """
        Get the forced alignment score for the given logits and text.

        Args:
            logits (np.ndarray): Logits from the ASR model
            texts (List[str]): List of texts to align
            sr (int): Sample rate of the audio

        Returns:
            tuple: List of scores and durations for best alignment of the text to the logits
        """
        # Get tokens for tests
        new_ids = [self.tokenizer.encode(text) for text in texts]

        # Get forced alignments
        scores, durations = [], []
        for new_id in new_ids:
            # Don't align empty sequences
            if new_id == []:
                scores.append(-100)
                durations.append(0)
                continue

            # Ensure that tokens are not longer than the time steps in the logits, otherwise truncate
            if len(new_id) >= logits.shape[0]:
                new_id = new_id[:logits.shape[0]-1]

            # Calll cpp forced align function
            t_labels, t_scores = forced_align(
                logits[None,],
                np.array(new_id)[None,],
                len(self.vocab)-1
            )
            t_labels = t_labels.flatten()

            # Get the average score and duration of the aligned text
            token_spans = self.merge_tokens(t_labels, t_scores)

            score = np.mean([i.score for i in token_spans if i.token != 1024])

            non_space_tokens = [i for i in token_spans if i.token != 1024]
            start = non_space_tokens[0].start*1280/sr
            end = non_space_tokens[-1].start*1280/sr

            durations.append(end - start)
            scores.append(score)

        return scores, durations

    def get_audio_features(self, audio: Union[str, np.ndarray], sr: int = 16000) -> tuple[np.ndarray, np.ndarray]:
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
                      approximate: bool = False,
                      ) -> tuple[List[str], List[float], List[float]]:
        """
        Match the intents for the given audio file or numpy array.

        Args:
            audio (Union[str, np.ndarray]): Audio file or numpy array of audio
            intents (List[str]): List of intents to search
            approximate (bool): If True, will use approximate intent similarities to more efficiently search for matching intents

        Returns:
            tuple: List of intents, scores, and durations
        """
        # Get the logits
        logits = self.get_logits(audio)

        # Get the intents

        if approximate is True and intents == []:
            raise ValueError("Approximate matching requires that intents be provided when initializing the model.")

        if approximate is True and intents != []:
            intents, scores, durations = self.match_intents_by_similarity(logits, self.build_intent_similarity_matrix(intents), intents)
        elif approximate is False and intents != []:
            intents, scores, durations = [], [], []
            for intent in intents:
                score, duration = self.get_forced_alignment_score(logits, [intent])
                intents.append(intent)
                scores.append(score[0])
                durations.append(duration[0])

        return intents, scores, durations