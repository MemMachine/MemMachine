#
# Stage 1: Builder
#
FROM python:3.12-slim-trixie AS builder

# Update OS and Python/PIP Packages
# Install curl
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip

# Copy uv binary from the source image INTO the builder stage
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

ARG SCM_VERSION="0.0.0"
ENV SETUPTOOLS_SCM_PRETEND_VERSION=${SCM_VERSION}

# Install dependencies into a virtual environment, but NOT the project itself.
# Every extra is installed except those for single-user pip installs: gpu
# (local models) and hnswlib (an index engine for the SQLite vector store).
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --package memmachine-server --no-install-workspace --no-editable --no-dev --all-extras --no-extra gpu --no-extra hnswlib

# Copy the application source code
COPY . /app

# Install the project itself from the local source
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --package memmachine-server --no-editable --no-dev --all-extras --no-extra gpu --no-extra hnswlib

#
# Stage 2: Final
#
FROM python:3.12-slim-trixie AS final

# Update OS and Python/PIP Packages
# Install curl
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip

WORKDIR /app

# Copy the environment from the builder stage
COPY --from=builder /app/.venv /app/.venv

# Set the PATH to include the virtual environment's bin directory
ENV PATH="/app/.venv/bin:$PATH"

# Download NLTK data and models
RUN python -c "import nltk, sys; \
    passed = nltk.download('punkt_tab') and nltk.download('stopwords'); \
    sys.exit(0 if passed else 1)"

# Set host to 0.0.0.0 to allow external access
ENV HOST=0.0.0.0

EXPOSE 8080
CMD ["sh", "-c", "memmachine-server"]
