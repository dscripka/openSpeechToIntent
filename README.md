# openSpeechToIntent

openSpeechToIntent is a library that maps audio containing speech to pre-specified lists of utterances. It can be used to directly map speech to categories and intents.

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

openSpeechToIntent is designed to be simple to use. Simply provide a file/array of audio data and a list of target intents, and the library will return information about potential matches.

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

TODO.

## Advanced Usage

TODO.

## Limitations

Currently, the library only supports matching English speech to english intents. Future work will focus on expanding the language and underlying ASR model support.

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.