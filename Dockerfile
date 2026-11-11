FROM python:3.13.9-slim-bookworm
WORKDIR /app
COPY main.py /app/main.py
RUN python -m py_compile /app/main.py
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
ENTRYPOINT ["python", "/app/main.py", "--data-dir", "/data"]
