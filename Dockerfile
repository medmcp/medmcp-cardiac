# syntax=docker/dockerfile:1
#
# medmcp-cardiac — cine cardiac MRI segmentation and ventricular function as a
# fixed-environment MCP stdio server. GPU (torch). Launched by the core via
# `docker run -i --device nvidia.com/gpu=all`.
#
# Every weight baked into this image is MIT (CineMA, Hugging Face mathpluscode/CineMA).
# The list comes from tools/checkpoints.py, which is also what the tools offer, so the
# image can never ship a different set.
ARG BASE_IMAGE=medmcp-base:dev
FROM ${BASE_IMAGE} AS runtime

# Stack metadata for one-click install/discovery (read via `docker inspect`, never
# by executing the image). tool_timeout_sec is kept in sync with server_config() and
# tools/segmentation.py (a test checks all three agree).
LABEL org.medmcp.stack='{"name": "medmcp-cardiac", "gpu": true, "tool_timeout_sec": 3600, "skills_path": "/app/src/medmcp_cardiac/skills"}'

# libgomp1 is needed by torch's OpenMP paths (CPU fallback).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Trust extra CA certs at build time behind a TLS-intercepting (MITM) proxy so
# uv/pip fetch through it. Drop the proxy root CA as a *.crt into ./certs/
# (gitignored; empty = no-op). UV_NATIVE_TLS makes uv use the system trust store.
# Runtime is offline, so no production impact.
COPY certs/ /usr/local/share/ca-certificates/medmcp-extra/
RUN update-ca-certificates
ENV UV_NATIVE_TLS=1

WORKDIR /app

# Dependencies first, from the committed lock, so a code change does not reinstall
# the multi-gigabyte torch/CUDA stack.
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project \
 && find /app/.venv -name '__pycache__' -type d -prune -exec rm -rf {} + \
 && find /app/.venv -name '*.a' -delete

# huggingface_hub uses requests, which trusts certifi's bundle rather than the
# system store, so it also needs pointing at the updated bundle to fetch through a
# MITM proxy. Harmless without a proxy CA.
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

# Weights live in a Hub cache at a fixed path rather than in a home directory, so
# they do not depend on which user the container runs as. Telemetry off before the
# first Hub call.
ENV HF_HOME=/opt/cinema \
    HF_HUB_DISABLE_TELEMETRY=1 \
    DO_NOT_TRACK=1

# Hugging Face intermittently answers 5xx. Retry with linear backoff, per file, so a
# failure re-fetches one checkpoint rather than restarting the build.
RUN printf '%s\n' '#!/bin/sh' \
      'n=0' \
      'until "$@"; do' \
      '  n=$((n+1))' \
      '  [ "$n" -ge 5 ] && { echo "retry: failed after $n attempts: $*" >&2; exit 1; }' \
      '  echo "retry: attempt $n failed; sleeping $((n*15))s" >&2' \
      '  sleep $((n*15))' \
      'done' \
    > /usr/local/bin/retry && chmod +x /usr/local/bin/retry

# Bake every checkpoint so segmentation runs with `--network none`. The file list
# comes from the same checkpoints module the tools use.
#
# Only checkpoints.py is copied in, ahead of the rest of the source, and loaded from
# that file directly (it imports nothing of ours). That keeps this layer's cache key
# to the one file that decides which weights are needed: with `COPY src` above it,
# editing any line of Python anywhere in the package would re-download 1.5 GB.
COPY src/medmcp_cardiac/tools/checkpoints.py /tmp/checkpoints_probe.py
RUN /app/.venv/bin/python -c \
        "import importlib.util as u; s = u.spec_from_file_location('checkpoints_probe', '/tmp/checkpoints_probe.py'); m = u.module_from_spec(s); s.loader.exec_module(m); print(m.HF_REPO_ID); print('\n'.join(m.weight_files()))" \
        > /tmp/weight_files.txt \
 && repo="$(head -n 1 /tmp/weight_files.txt)" \
 && echo "baking $(( $(wc -l < /tmp/weight_files.txt) - 1 )) files from ${repo}" \
 && tail -n +2 /tmp/weight_files.txt | while read -r f; do \
        echo "--- ${f}"; \
        retry /app/.venv/bin/python -c \
            "import sys; from huggingface_hub import hf_hub_download as d; d(repo_id=sys.argv[1], filename=sys.argv[2])" \
            "${repo}" "${f}"; \
    done \
 && rm -f /tmp/weight_files.txt /tmp/checkpoints_probe.py \
 && find "${HF_HOME}" -name '*.lock' -delete

# From here on the Hub is never contacted: hf_hub_download resolves every file from
# the cache above and raises, rather than connecting, if one is missing. The core
# runs the container with --network none anyway; this makes a cache miss a clear
# error instead of a hang.
ENV HF_HUB_OFFLINE=1

# The source, and the project itself, last: everything above depends only on the
# lock and on checkpoints.py, so a code change rebuilds seconds of work instead of
# re-fetching the weights.
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --no-deps --python /app/.venv/bin/python . \
 && find /app/.venv -name '__pycache__' -type d -prune -exec rm -rf {} +

ENV PATH=/app/.venv/bin:$PATH \
    UV_NO_SYNC=1

# stdio MCP server. tini reaps the process and forwards signals; stdio passes through.
ENTRYPOINT ["tini", "--", "medmcp-cardiac"]
