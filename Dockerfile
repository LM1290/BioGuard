
FROM --platform=linux/amd64 mambaorg/micromamba:1.5.3

WORKDIR /app

COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/env.yaml


RUN micromamba install -y -n base -f /tmp/env.yaml && \
    micromamba clean --all --yes


ARG MAMBA_DOCKERFILE_ACTIVATE=1


RUN pip install --no-cache-dir torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir \
    torch-scatter \
    torch-sparse \
    torch-cluster \
    torch-spline-conv \
    torch-geometric \
    -f https://data.pyg.org/whl/torch-2.1.0+cpu.html


COPY --chown=$MAMBA_USER:$MAMBA_USER . .

EXPOSE 8000

ENTRYPOINT ["/usr/local/bin/_entrypoint.sh", "python", "-m", "bioguard.main", "serve"]
