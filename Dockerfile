FROM --platform=linux/amd64 python:3.6

WORKDIR /app

# System libs the old Unity (Mono) headless binary may need
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Pinned ML-Agents v0.4 communication stack + agent deps.
# protobuf < 3.20 is required for the old *_pb2.py generated files.
RUN pip install --no-cache-dir \
        numpy==1.19.5 \
        protobuf==3.6.1 \
        grpcio==1.11.0 \
        pyyaml \
        docopt \
        pillow

# PyTorch (CPU) for the DDPG agent + plotting deps
RUN pip install --no-cache-dir torch==1.10.2+cpu \
        -f https://download.pytorch.org/whl/torch_stable.html && \
    pip install --no-cache-dir matplotlib pandas

COPY . /app

CMD ["python", "test_env.py"]
