# openSpeechToIntent

openSpeechToIntent is a library that maps audio containing speech to pre-specified lists of utterances. It can be used to directly map speech to categories and intents.

This is accomplished by using small but robust speech-to-text models (the default is [Citrinet 256 by NVIDIA](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/stt_en_citrinet_256)) to generate predictions for characters or tokens, and then using [forced alignment](https://research.nvidia.com/labs/conv-ai/blogs/2023/2023-08-forced-alignment/) to determine if the audio matches different text transcriptions. The score of the alignment is used to determine which (if any) of the transcriptions is a good match to the audio.

## Installation

To install openSpeechToIntent, you can simply use pip:

```bash
pip install openSpeechToIntent
```

This should work on nearly all systems, as there are only three requirements: `numpy`, `onnxruntime`, and `pybind11`.

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.