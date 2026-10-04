#!/bin/bash
# Run one capture script inside the pinned TensorRT-LLM image. Never starts a model.
#
#   run.sh <script.py> [args...]      (the script sits next to this file and is mounted read-only at /work)
#
# Safety: no network, all capabilities dropped, container removed on exit, exactly ONE GPU attached (the card with the
# most free memory, because the engine's Python package refuses to import without a CUDA driver library; it allocates
# no GPU memory for any script here). Output of the script is on stdout; NVIDIA's entrypoint prints a licence banner
# first, so consumers cut everything before the first line that is exactly "{" or "[".
set -eu
IMAGE="${TRTLLM_IMAGE:-nvcr.io/nvidia/tensorrt-llm/release:1.2.1}"
HERE="$(cd "$(dirname "$0")" && pwd)"
GPU="$(nvidia-smi --query-gpu=uuid,memory.free --format=csv,noheader,nounits | sort -t, -k2 -n -r | head -1 | cut -d, -f1 | tr -d ' ')"
[ -n "$GPU" ] || { echo "no GPU found" >&2; exit 2; }
SCRIPT="$1"
shift
exec docker run --rm --network none --cap-drop ALL --gpus "device=${GPU}" -v "${HERE}:/work:ro" "$IMAGE" \
  python3 "/work/${SCRIPT}" "$@"
