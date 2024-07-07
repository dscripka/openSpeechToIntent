# openSpeechToIntent

openSpeechToIntent is a library that maps audio containing speech to pre-specified lists of short texts. It can be used to directly map speech to these texts, which can represent, categories or intents for various voice automation applications.

This is accomplished by using small but robust speech-to-text models (the default is [Citrinet 256 by NVIDIA](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/stt_en_citrinet_256)) to generate predictions for characters or tokens, and then using [forced alignment](https://research.nvidia.com/labs/conv-ai/blogs/2023/2023-08-forced-alignment/) to determine if the audio matches different text transcriptions. The score of the alignment is used to determine which (if any) of the transcriptions is a good match to the audio.

This approach has some nice advantages:
- No training is required, and any text can be matched to any audio
- It is relatively fast on a wide range of hardware (see the [Performance](#performance) for more details), as the underlying models are small and efficient
- It can be easily combined with other libraries (e.g., [openWakeWord](https://github.com/dscripka/openwakeword)) and tools to create more complex audio processing pipelines

## Installation

To install openSpeechToIntent, you can simply use pip:

```bash
pip install openSpeechToIntent
```

This should work on nearly all systems, as there are only three requirements: `numpy`, `onnxruntime`, and `pybind11`.

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
audio_file = "test/resources/sample_1.wav"  # contains speech "turn on the light"

# Match the audio to the provided intents
matched_intents, scores, durations = mdl.match_intents(audio_file, intents)

# View the results
for intent, score, duration in zip(matched_intents, scores, durations):
    print(f"Intent: {intent}, Score: {score}, Duration: {duration}")

# Output:
# Intent: turn on the light, Score: -0.6, Duration: 0.8
# Intent: pause the music, Score: -5, Duration: 0.5
# Intent: set a 5 minute timer, Score: -6, Duration: 1.6
```

Scores are from the un-modified logits, so smaller scores (closer to zero) are better. The durations are the approximate length of the intent as aligned to the audio. This can provide another way to filter out bad matches, by ignoring those that seem too short or too long for the intent.

## Performance

For many use-cases, the performance of openSpeechToIntent can be surprisingly good. This is a testament to both the high quality of Nvidia pre-trained models, and the way that constraining the ASR decoding to a fixed set of intents greatly reduces the search space of the problem. While real-world performance numbers always depend on the deployment environment, here are some examples use-cases that illustrate the type of performance possible with openSpeechToIntent:

## Ignoring false wake word activations

Wake word detection frameworks like [openWakeWord](), [microWakeWord](), etc. are designed to efficiently listen for target activation words will continuously processing input audio. The challenge with these types of systems is maintaining high recall of the target activation words, while not activating on other, unrelated audio. In practice, this is a difficult balance that requires careful training and tuning, and performance can vary widely depending on the environment and the specific wakeword.

One approach to improving the effective performance of these types of systems is to tune the wakeword system to be very sensitive, and filter out any false activations through other means. openSpeechToIntent can be used in this way, assuming that there is a known list of intents that would normally be expected after a wake word activation. As an example, the table below shows the performance of the pre-trained `Alexa` wake word model from openWakeWord on the 24 hour [PicoVoice wake word benchmark dataset](), where the model's threshold is set very low (0.1) to ensure high recall but low precision. With this configuration, the false positive rate of X is unnacceptably high. However, by using openSpeechToIntent to very that the speech after the activation matches a list ~400 expected intents (see the list [here]()), this rate can be reduced by over x%, to a satisfactory level of x per hour.

TODO: Add table here, and include the threshold level set for the forced alignment score

## Precisely matching a small number of intents

openSpeechToIntent can also be used to perform more fine-grained classification. As an example, on the [Fluent Speech Commands]() test set, an accuracy of x% across 35 intents is possible.

Use the fluent speech commands dataset and check accuracy.

## Matching a large number of intents

Use synthetic dataset of 400+ intents (and maybe make it larger?) and show the accuracy of the model across all intents.

## Advanced Usage

TODO.

## Limitations

Currently, the library only supports matching English speech to english intents. Future work will focus on expanding the language and underlying ASR model support.

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.