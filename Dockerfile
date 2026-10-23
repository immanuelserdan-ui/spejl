# syntax=docker/dockerfile:1
# RapidOCR's current ONNX runtime package requires Python <3.13.
FROM python:3.12-slim-bookworm@sha256:34386ef0cb081344d7ec1c103ba398e6e9f64e9ab3a1509accc92a4e24a07258 AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH="/opt/venv/bin:$PATH"

# OpenCV/ONNX native libraries and the fonts Spejl resolves on Linux.
RUN apt-get update && apt-get install -y --no-install-recommends \
        fonts-dejavu-core \
        fonts-liberation \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

FROM base AS builder
WORKDIR /build
RUN python -m venv /opt/venv
COPY pyproject.toml README.md ./
COPY spejl/ ./spejl/
RUN python -m pip install --no-cache-dir ".[raster]"

FROM base AS cli
COPY --from=builder /opt/venv /opt/venv
COPY docker/smoke_test.py /opt/spejl-smoke.py
RUN groupadd --gid 10001 spejl \
    && useradd --uid 10001 --gid spejl --create-home spejl \
    && mkdir /work && chown spejl:spejl /work
USER 10001:10001
WORKDIR /work
ENTRYPOINT ["spejl"]
CMD ["--help"]

# Optional development target. Qt is installed only here for GUI unit tests.
FROM cli AS test
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
        libdbus-1-3 \
        libegl1 \
        libfontconfig1 \
        libxkbcommon0 \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip install --no-cache-dir "spejl[gui,dev]"
WORKDIR /tests
COPY pyproject.toml ./
COPY tests/ ./tests/
COPY tools/ ./tools/
COPY installer/Spejl.iss ./installer/Spejl.iss
ENV QT_QPA_PLATFORM=offscreen
USER 10001:10001
ENTRYPOINT ["python", "-m", "pytest"]
CMD ["-m", "not slow", "-q", "-p", "no:cacheprovider", "--basetemp=/tmp/pytest"]

# Keep the default docker build output as the CLI image, without test/Qt tools.
FROM cli AS runtime
