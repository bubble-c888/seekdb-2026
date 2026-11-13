# MiniVectorDB - evaluation image
#
# The platform builds this image and starts the program in the foreground:
#
#   docker build -t minivectordb .
#   docker run -i --rm -v /host/data:/data minivectordb
#
# The test driver talks to the container over stdin/stdout, therefore no port
# is exposed and no TTY is allocated.  All database files live in /data.
#
# The dependencies are installed while the image is built; the container itself
# never needs network access.

FROM python:3.13-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    MINIVECTORDB_DATA_DIR=/data \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

# numpy is the only third party dependency: it provides the float32 arithmetic
# of the vector index and the vectorised scans.  The version is pinned.
RUN pip install --no-cache-dir "numpy==2.5.3"

WORKDIR /app

# The database is a pure Python package, so the sources are copied as they are.
COPY minivectordb/ /app/minivectordb/
COPY tools/ /app/tools/
COPY protocol_check/ /app/protocol_check/
COPY mysqltest/ /app/mysqltest/
COPY README.md /app/README.md

RUN mkdir -p /data

# Stays in the foreground, reads SQL from stdin and writes the results to
# stdout.  "python -u" keeps stdout unbuffered, the protocol layer flushes
# after every statement as well.
ENTRYPOINT ["python", "-u", "-m", "minivectordb", "--data-dir", "/data"]
