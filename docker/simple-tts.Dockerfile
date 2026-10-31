FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 build-essential && rm -rf /var/lib/apt/lists/*

RUN uv pip install --system --no-cache gradio vosk-tts edge-tts scipy numpy soundfile miniaudio

WORKDIR /app
COPY simple_tts_server.py .

EXPOSE 22231

CMD ["python", "-c", "from simple_tts_server import build_and_launch; build_and_launch(server_name='0.0.0.0')"]
