# Build from the repository root so the AIAssistant/ prefix resolves:
#   docker build -f AIAssistant/Dockerfile -t ai-assistant .
# The image runs the platform entry point; it does NOT ship model weights
# (mount them at /app/AIAssistant/Models or set APP_DATA_DIR).
FROM python:3.11-slim

WORKDIR /app

# Dependency set first so source changes do not invalidate the wheel layer.
COPY AIAssistant/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY AIAssistant ./AIAssistant

# P0: the image must start the package it just copied. Editable install keeps
# ``__file__`` inside /app/AIAssistant, so config/ and logs/ resolve exactly as
# they do in a source checkout (a plain install would relocate them into
# site-packages and drop config/*.toml on the floor).
RUN pip install --no-cache-dir -e ./AIAssistant

ENV AGENT_OBSERVABILITY=1
ENV PYTHONUNBUFFERED=1
# Data (models, RAG cache, task store, logs) goes to a mountable volume instead
# of wherever the code happens to live.
ENV APP_DATA_DIR=/app/data
# Loopback-only binding makes the port unreachable from outside the container;
# ADR 0008 still gates the agent/admin/chat surfaces to loopback callers, so
# exposing the interface does not open the mutating APIs.
ENV AGENT_HOST=0.0.0.0
ENV AGENT_PORT=8080

EXPOSE 8080

# python:3.11-slim has no curl — probe with the stdlib instead.
# /readyz answers 503 while the LLM is not loaded, 200 when it is, so a
# first-boot model download shows up as "starting" rather than a dead service.
# start-period is generous: a cold index build + model load measured ~337s.
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=3 \
  CMD python -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/readyz', timeout=4).status == 200 else 1)"

CMD ["python", "-m", "ai_assistant.bootstrap.server"]
