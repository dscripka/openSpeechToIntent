# openSpeechToIntent

openSpeechToIntent is a library that maps audio containing speech to pre-specified lists of short texts. It can be used to directly map speech to these texts, which can represent, categories or intents for various voice automation applications.

This is accomplished by using small but robust speech-to-text models (the default is [Citrinet 256 by NVIDIA](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/stt_en_citrinet_256)) to generate predictions for characters or tokens, and then using [forced alignment](https://research.nvidia.com/labs/conv-ai/blogs/2023/2023-08-forced-alignment/) to determine if the audio matches different text transcriptions. The score of the alignment is used to determine which (if any) of the transcriptions is a good match to the audio.

This approach has several useful features:
- No training is required, and any text can be matched to any audio
- It has strong performance while still being relatively fast on a wide range of hardware (see the [Performance](#performance) for more details), as the underlying models are small and efficient
- It can be easily combined with other libraries (e.g., [openWakeWord](https://github.com/dscripka/openwakeword)) and tools to create more complex audio processing pipelines

## Installation

To install openSpeechToIntent, you can simply use pip:

```bash
pip install openSpeechToIntent
```

This should work on nearly all operating systems (Windows, macOS, Linux), as there are only three requirements: `numpy`, `onnxruntime`, and `pybind11`.

## Usage

openSpeechToIntent is designed to be simple to use. Simply provide a file/array of audio data and a list of target intents, and the library will return information about potential intent matches.

```python

from openspeechtointent.model import CitrinetModel

# Load model
mdl = CitrinetModel()

# Define some simple intents
intents = ["turn on the light", "pause the music", "set a 5 minute timer"]

# Load a sample audio file (from the test/resources directory in this repo)
# Can also directly provide a numpy array of 16-bit PCM audio data
audio_file = "test/resources/sample_1.wav"  # contains the speech "turn on the lights"

# Match the audio to the provided intents
matched_intents, scores, durations = mdl.match_intents(audio_file, intents)

# View the results
for intent, score, duration in zip(matched_intents, scores, durations):
    print(f"Intent: {intent}, Score: {score}, Duration: {duration}")

# Output:
# Intent: "turn on the lights", Score: 0.578, Duration: 0.800 seconds
# Intent: "pause the music", Score: 0.270, Duration: 0.560 seconds
# Intent: "set a 5 minute timer", Score: 0.119, Duration: 0.560 seconds
# Intent: "remind me to buy apples tomorrow", Score: 0.032, Duration: 1.840 seconds
```

Scores are computed from the softmaxed logits from the Citrinet model, to scale them between 0 and 1. The score can be used to select possible matching intents (or none at all) by appropriate thresholds.

The durations are the approximate length of the intent as aligned to the audio. This can provide another way to filter and select possible intent matches by selecting those that have the most appropriate duration.

## Performance

For many use-cases, the performance of openSpeechToIntent can be surprisingly good. This is a testament to both the high quality of Nvidia pre-trained models, and the way that constraining the speech-to-text decoding to a fixed set of intents greatly reduces the search space of the problem. While real-world performance numbers always depend on the deployment environment, here are some examples use-cases that illustrate the type of performance possible with openSpeechToIntent:

### Ignoring false wake word activations

Wake word detection frameworks like [openWakeWord](https://github.com/dscripka/openWakeWord), [microWakeWord](https://github.com/kahrendt/microWakeWord), etc. are designed to efficiently listen for target activation words while continuously processing input audio. The challenge with these types of systems is maintaining high recall of the target activation words, while not activating on other, unrelated audio. In practice, this is a difficult balance that requires careful training and tuning, and performance can vary widely depending on the environment and the specific wake word.

One approach to improving the effective performance of these types of systems is to tune the wakeword model to be very sensitive, and then filter out any false activations through other means. openSpeechToIntent can be used in this way, assuming that there is a known list of intents that would normally be expected after a wake word activation. As an example, the table below shows the performance of the pre-trained `Alexa` wake word model from openWakeWord on the 24 hour [PicoVoice wake word benchmark dataset](https://github.com/Picovoice/wake-word-benchmark), where the model's threshold is set very low (0.1) to ensure very high recall but low precision. With this configuration, false positive rate on the Picovoice dataset is ~3.58, unnacceptably high. However, by using openSpeechToIntent to verify that the speech after the activation matches a list ~400 expected intents (see the list [here]()), the false positive rate can be reduced to <0.04 false activations per hour.

| openWakeWord Model | openWakeWord Score Threshold | openSpeechToIntent Score Threshold | False Positives per Hour |
|---------------------|------------------------------|------------------------------------|--------------------------|
| Alexa              | 0.1                          | NA                                 | ~3.58                     |
| Alexa              | 0.1                          | 0.1                                 | <0.04                     |


### Precisely matching a small number of intents

openSpeechToIntent can also be used to perform more fine-grained classification. As an example, on the [Fluent Speech Commands](https://fluent.ai/fluent-speech-commands-a-dataset-for-spoken-language-understanding-research/) test set, an average accuracy of ~98% across 31 intents is possible.

| Model | Accuracy |
|-------|----------|
| openSpeechToIntent | 98.2% |
| [SOTA](https://paperswithcode.com/paper/finstreder-simple-and-fast-spoken-language) | 99.7% |


### Matching a large number of intents

Use synthetic dataset of 400+ intents (and maybe make it larger?) and show the accuracy of the model across all intents.

### Efficiency

openSpeechToIntent is designed to be reasonably efficient, and can run on a wide range of hardware included normal desktop CPUs and moderately powerfull SBCs. The table below shows the performance of the default Nvidia Citrinet model on a several different systems, using a 4 second audio clip as input.

| CPU | Number of Threads | Time to Process 4s Audio Clip (ms) | Time to Match Against 100 Intents (ms) |
|-----|-------------------|-------------------------------|---------------------------|
| Intel Xeon W-2123 | 1 | 103 | 20 |
| AMD Ryzen 1600 | 1 | 98 | 27 |
| Raspberry Pi 4 | 1 | 320 | 110 |
| Raspberry Pi 4 | 2 | 262 | 110 |

Note that further optimizations are possible (see the [Advanced Usage](#advanced-usage) section), and in general the length of the audio clip
and the number of intents will have the largest impact on the efficiency of the system.

## Advanced Usage

### Limited the search over intents

If the number of intents that you want to match against is very large, this can lead to low efficiency in some devices as it is difficult (currently) to parallelize the search across all intents. However, because in many cases some intents will be very similar (by textual overlap) to other intents, heuristics can be used to limit the search space to only broadly unique intents. openSpeechtoIntent provides on such function that implements this heuristic:

```python

```

### Using raw logit scores

TODO

## Limitations

Currently, the library only supports matching English speech to english intents. Future work may involve expanding to other languages and supporting other speech-to-text frameworks like [whisper.cpp](https://github.com/ggerganov/whisper.cpp).

## Acknowledgements

Many thanks to Nvidia for the excellent Citrinet speech-to-text models, as well as many other highly performant speech and audio models.

Also, credit to @MahmoudAshraf97 for the excellent modification of the [torch forced alignment cpp functions](https://github.com/MahmoudAshraf97/ctc-forced-aligner/blob/main/ctc_forced_aligner/forced_align_impl.cpp) to simplify dependencies and enable simple usage with `pybind`.

## License

This code in this project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details. Portions of the code adapted in whole or part from other repositories are licensed under their respective licenses, as appropriate.

The Nvidia Citrinit models is licensed under the [CC-BY-4.0 license](https://creativecommons.org/licenses/by/4.0/) and the NGC [Terms of Use](https://ngc.nvidia.com/legal/terms).