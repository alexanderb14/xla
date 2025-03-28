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
# -o history: record shell history
set -euox pipefail -o history

# Generate a templated results file to make output accessible to everyone
"$KOKORO_ARTIFACTS_DIR"/github/xla/.kokoro/generate_index_html.sh "$KOKORO_ARTIFACTS_DIR"/index.html

function is_linux_gpu_job() {
  [[ "$KOKORO_JOB_NAME" =~ tensorflow/xla/linux/.*gpu.* ]]
}

# Pull the container (in case it was updated since the instance started) and
# store its SHA in the Sponge log.
docker pull "$DOCKER_IMAGE"
echo "TF_INFO_DOCKER_IMAGE,$DOCKER_IMAGE" >> "$KOKORO_ARTIFACTS_DIR/custom_sponge_config.csv"
echo "TF_INFO_DOCKER_SHA,$(docker pull "$DOCKER_IMAGE" | sed -n '/Digest:/s/Digest: //g p')" >> "$KOKORO_ARTIFACTS_DIR/custom_sponge_config.csv"

# Start a container in the background
docker run --name xla -w /tf/xla -itd --rm \
    -v "$KOKORO_ARTIFACTS_DIR/github/xla:/tf/xla" \
    -v "$KOKORO_ARTIFACTS_DIR/pkg:/tf/pkg" \
    "$DOCKER_IMAGE" \
    bash

# bazelrc Files currently come from https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/tf_sig_build_dockerfiles/devel.usertools
RC_FILE="/usertools/cpu.bazelrc"
TARGET_FILTER="-//xla/hlo/experimental/... -//xla/python_api/... -//xla/python/..."
TAGS_FILTER="-no_oss,-oss_serial"
ADDITIONAL_FLAGS=""

if is_linux_gpu_job ; then
    TAGS_FILTER="$TAGS_FILTER,gpu,requires-gpu,-no_gpu"
    ADDITIONAL_FLAGS="$ADDITIONAL_FLAGS --run_under=//tools/ci_build/gpu_build:parallel_gpu_execute"
    RC_FILE="/usertools/gpu.bazelrc"
    # disable three tests that fail to build at the moment (b/263149095)
    TARGET_FILTER="$TARGET_FILTER -//xla/service/gpu:gpu_device_info_test -//xla/stream_executor/cuda:cuda_driver_test_cpu -//xla/stream_executor/cuda:cuda_driver_test_gpu"
else
    TAGS_FILTER="$TAGS_FILTER,-gpu,-requires-gpu"
fi

# Build & test XLA
docker exec xla bazel --bazelrc=$RC_FILE \
        test \
        --build_tag_filters=$TAGS_FILTER  \
        --test_tag_filters=$TAGS_FILTER \
        --nocheck_visibility \
        --keep_going \
        --profile=/tf/pkg/profile.json.gz \
        --config=nonccl \
        --flaky_test_attempts=3 \
        --config=rbe \
        --jobs=150 \
        $ADDITIONAL_FLAGS \
        -- //xla/... $TARGET_FILTER |& grep -v "violates visibility of"
# Print build time statistics, including critical path.
docker exec xla bazel analyze-profile "/tf/pkg/profile.json.gz"
# Stop container
docker stop xla


