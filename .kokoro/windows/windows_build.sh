#!/bin/bash
# Copyright 2022 Google LLC All Rights Reserved.
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

# -e: abort script if one command fails
# -u: error if undefined variable used
# -o pipefail: entire command fails if pipe fails. watch out for yes | ...
# Note: set -x <code> +x around anything you want to have logged.
set -euo pipefail

cd "${KOKORO_ARTIFACTS_DIR}/github/xla"

export PATH="$PATH:/c/Python38"

TARGET_FILTER=-//xla/hlo/experimental/... -//xla/python_api/... -//xla/python/...

/c/tools/bazel.exe build \
  --output_filter="" \
  --nocheck_visibility \
  --keep_going \
  -- //xla/... $TARGET_FILTER \
  || { exit 1; }

exit 0
