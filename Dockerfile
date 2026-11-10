FROM python:3.12.16-slim-trixie
WORKDIR /app
COPY main.py engine.py vector_index.py storage.py ./
ENTRYPOINT ["python", "-u", "/app/main.py"]
