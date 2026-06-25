ARG BASE_IMAGE=pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
FROM ${BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app

COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

COPY README.md ./
COPY src ./src
COPY experiments ./experiments
COPY scripts ./scripts

RUN mkdir -p data results figures analysis

CMD ["python", "-m", "experiments.run_sa"]
