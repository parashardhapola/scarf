FROM ubuntu:20.04 AS cilk-runtime

# The bundled sgtsne executable requires the Cilk runtime from Ubuntu 20.04.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libcilkrts5 \
    && rm -rf /var/lib/apt/lists/*

FROM ubuntu:22.04

COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /uvx /usr/local/bin/
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential ca-certificates libfftw3-double3 libmetis5 libtbbmalloc2 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=cilk-runtime /usr/lib/x86_64-linux-gnu/libcilkrts.so.5* /usr/local/lib/
COPY --from=cilk-runtime /usr/share/doc/libcilkrts5/copyright /usr/share/doc/libcilkrts5/copyright
RUN ldconfig

WORKDIR /workspace
COPY pyproject.toml uv.lock setup.py VERSION requirements.txt requirements_extra.txt requirements_harmony.txt requirements_leiden.txt pypi_README.rst LICENSE THIRD_PARTY_NOTICES.md .python-version ./
COPY licenses ./licenses
COPY bin/sgtsne ./bin/sgtsne
COPY scarf/*.py ./scarf/
COPY scarf/datastore/*.py ./scarf/datastore/
RUN uv sync --locked --no-editable --extra extra --extra harmony --extra leiden --python 3.12 \
    && uv cache clean

ENV PATH="/workspace/.venv/bin:$PATH"
