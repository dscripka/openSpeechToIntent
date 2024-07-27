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

# imports
import os
import urllib.request
import sys

def download_file(url, target_directory, file_size=None):
    """A simple function to download a file from a URL with a progress bar using only standard libraries."""
    local_filename = url.split('/')[-1]
    file_path = os.path.join(target_directory, local_filename)

    # Open the URL
    with urllib.request.urlopen(url) as response:
        if file_size is None:
            file_size = int(response.getheader('Content-Length', 0))
        
        # Create a progress bar
        print(f"Downloading {local_filename} ({file_size} bytes)")
        downloaded = 0
        
        with open(file_path, 'wb') as f:
            while True:
                chunk = response.read(8192)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                
                # Update progress
                progress = downloaded / file_size * 100 if file_size else 0
                sys.stdout.write(f"\rProgress: {progress:.2f}%")
                sys.stdout.flush()
