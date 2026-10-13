FROM python:3.11-slim
WORKDIR /app
COPY AIAssistant/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY AIAssistant ./AIAssistant
ENV AGENT_OBSERVABILITY=1
ENV PYTHONUNBUFFERED=1
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=60s --retries=3 CMD curl -f http://localhost:8000/readyz || exit 1
CMD ["python", "-m", "ai_assistant.bootstrap.server"]
