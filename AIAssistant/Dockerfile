# Build from the repository root so the AIAssistant/ prefix resolves:
#   docker build -f AIAssistant/Dockerfile -t ai-assistant .
# The image runs the platform entry point; it does NOT ship model weights
# (mount them at /app/AIAssistant/Models or set APP_DATA_DIR).
#
# Multi-stage (10/10 hardening): the builder compiles/installs wheels, the
# runtime carries only the installed site-packages + source tree, run as a
# non-root user so a tool-execution escape does not inherit uid 0.
#
# ADR 0032 — three reproducibility defects closed:
#   * the base image is pinned by digest, not by a floating tag;
#   * the image installs the *hash-pinned* lock, not the floating ranges;
#   * the health probe checks liveness, not readiness — the image ships no
#     model weights, so ``/readyz`` answers 503 forever and the container
#     would be permanently "unhealthy" while actually serving fine.
FROM python:3.11-slim@sha256:e529028263dbe6910a2d96f7d2b8f5266385e917fd45d286ef166977c094a51e AS builder

WORKDIR /build

# Dependency set first so source changes do not invalidate the wheel layer.
# The container lock is hash-pinned and CPU-only: the runtime lock resolves
# faiss-gpu for every non-Windows platform, which would pull a CUDA wheel
# into a slim CPU image. pip refuses any artifact whose SHA-256 does not
# match, so a republished or tampered wheel cannot enter the image.
COPY AIAssistant/requirements.txt AIAssistant/requirements.lock \
     AIAssistant/requirements-docker.txt AIAssistant/requirements-docker.lock ./
RUN pip install --no-cache-dir --prefix=/install --require-hashes -r requirements-docker.lock

COPY AIAssistant ./AIAssistant
# Editable install keeps ``__file__`` inside /app/AIAssistant, so config/ and
# logs/ resolve exactly as they do in a source checkout (a plain install would
# relocate them into site-packages and drop config/*.toml on the floor).
# Installed into the same prefix so the runtime copies one tree.
RUN pip install --no-cache-dir --prefix=/install -e ./AIAssistant


FROM python:3.11-slim@sha256:e529028263dbe6910a2d96f7d2b8f5266385e917fd45d286ef166977c094a51e AS runtime

WORKDIR /app

COPY --from=builder /install /usr/local
COPY --from=builder /build/AIAssistant ./AIAssistant
# The editable-install finder points back at /build/AIAssistant; re-install
# the pointer against the runtime path (no network, no deps re-resolve).
RUN pip install --no-cache-dir -e ./AIAssistant --no-deps

ENV AGENT_OBSERVABILITY=1
ENV PYTHONUNBUFFERED=1
# Data (models, RAG cache, task store, logs) goes to a mountable volume instead
# of wherever the code happens to live.
ENV APP_DATA_DIR=/app/data
# The container is reachable from the network by design, so the bind address
# is 0.0.0.0. That does NOT open the mutating APIs: ADR 0008 gates the
# agent/admin/chat surfaces to loopback callers, and the OpenAPI surface is
# not published at all unless the deployment opts in (ADR 0032).
ENV AGENT_HOST=0.0.0.0
ENV AGENT_PORT=8080
ENV PATH="/usr/local/bin:${PATH}"
ENV PYTHONPATH="/app/AIAssistant/src:${PYTHONPATH}"

# Non-root runtime. ADR 0032: the user creation and the chown are two steps,
# and neither is allowed to fail silently. The previous one-liner was
# ``a && b && c || true``, which in shell succeeds when *a* fails — a failed
# ``useradd`` produced a green build and a container that died at
# ``USER appuser``.
RUN useradd --create-home --shell /usr/sbin/nologin appuser
RUN mkdir -p /app/data /app/AIAssistant/logs \
    && chown -R appuser:appuser /app \
    && chown -R appuser:appuser /usr/local/lib/python3.11/site-packages/__editable__* \
    || (echo "chown failed" >&2; exit 1)

EXPOSE 8080

# python:3.11-slim has no curl — probe with the stdlib instead.
# /livez answers 200 as soon as the event loop answers; /readyz answers 503
# while the LLM is not loaded. The image ships no weights, so the readiness
# probe would never pass: the liveness probe is the one that describes the
# container. A deployment that mounts a model and wants readiness gating
# should probe /readyz itself.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/livez', timeout=4).status == 200 else 1)"

USER appuser

CMD ["python", "-m", "ai_assistant.bootstrap.server"]
