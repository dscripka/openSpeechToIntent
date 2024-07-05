# TODO: Add license and disclosures of the source of the original code (mostly torchaudio)

import numpy as np
from typing import List, NamedTuple
import json
import pickle
from typing import List, Union
import onnxruntime as ort
import scipy
import torch
import torchaudio

class TokenSpan(NamedTuple):
    token: int
    start: int
    end: int
    score: float

class CitrinetModel:
    def __init__(self, model_path: str="resources/models/stt_en_citrinet_256.onnx", ncpu: int=1):
        """
        Initialize the Citrinet model, including pre-processing functions.
        Model is obtained from https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/stt_en_citrinet_256
        and then converted to the ONNX format using the standard Nvidia NeMo tools.

        Args:
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
        self.stft = ort.InferenceSession("resources/models/torchlibrosa_stft.onnx", sess_options=sess_options)

        # Load filterbank
        self.filterbank = np.load("resources/models/citrinet_spectrogram_filterbank.npy")

        # Load tokenizer and vocab
        self.tokenizer = pickle.load(open("resources/models/tokenizer.pkl", "rb"))
        self.vocab = json.load(open("resources/models/citrinet_vocab.json", 'r'))

        # Load intents
        self.intents = json.load(open("resources/intents/default_intents.json", 'r'))["smart_home_intents"]

    def get_seq_len(self, seq_len):
        pad_amount = 512 // 2 * 2
        seq_len = np.floor_divide((seq_len + pad_amount - 512), 160) + 1
        return seq_len.astype(np.int64)

    def normalize_batch(self, x, seq_len, normalize_type):
        x_mean = None
        x_std = None
        if normalize_type == "per_feature":
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

    def get_features(self, x, length):
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
        x, _, _ = self.normalize_batch(x, seq_len, normalize_type="per_feature")

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

    def get_forced_alignment_score(self, logits, texts, sr = 16000):
        # Get tokens for tests
        new_ids = [self.tokenizer.encode(text) for text in texts]

        # Get forced alignments
        scores, durations = [], []
        for new_id in new_ids:
            t_labels, t_scores = torchaudio.functional.forced_align(
                log_probs = torch.from_numpy(logits[None,]),
                targets = torch.from_numpy(np.array(new_id)[None,]),
                blank = len(self.vocab)-1
            )

            t_labels, t_scores = t_labels[0].numpy(), t_scores[0].numpy()

            # Get the average score and duration of the aligned text
            token_spans = self.merge_tokens(t_labels, t_scores)

            score = np.mean([i.score for i in token_spans if i.token != 1024])

            non_space_tokens = [i for i in token_spans if i.token != 1024]
            start = non_space_tokens[0].start*1280/sr
            end = non_space_tokens[-1].start*1280/sr

            durations.append(end - start)
            scores.append(score)

        return scores, durations

    def get_audio_features(self, audio: Union[str, np.ndarray], sr: int = 16000):
        if isinstance(audio, str):
            sr, wav_dat = scipy.io.wavfile.read(audio)
        else:
            wav_dat = audio

        # Convert to float32 from 16-bit PCM
        wav_dat = (wav_dat/32767).astype(np.float32)
        wav_dat = np.pad(wav_dat, (4000, 4000), mode='constant')
        all_features, lengths = self.get_features(wav_dat[None,], np.array([wav_dat.shape[0]]))

        return all_features, lengths

    def get_logits(self, audio: Union[str, np.ndarray]):
        # Preprocess audio
        all_features, lengths = self.get_audio_features(audio)

        # Transcribe processed audio with the onnx model
        logits = self.asr_model.run(None, {self.asr_model.get_inputs()[0].name: all_features.astype(np.float32), "length": lengths})
        logits = logits[0][0]

        return logits
