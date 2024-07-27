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


import os

MODELS = {
    "stft": {
        "model_path": os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources/models/torchlibrosa_stft.onnx"),
        "download_url": "https://github.com/dscripka/openSpeechtoIntent/releases/download/v0.1.0.alpha/torchlibrosa_stft.onnx"
    },
    "citrinet_256": {
        "model_path": os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources/models/stt_en_citrinet_256.onnx"),
        "download_url": "https://github.com/dscripka/openSpeechtoIntent/releases/download/v0.1.0.alpha/stt_en_citrinet_256.onnx"
    }
}