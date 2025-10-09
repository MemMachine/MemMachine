#
# Stage 1: Builder
#
FROM python:3.12-slim-trixie AS builder

# Update OS and Python/PIP Packages
# Install curl
RUN << EOF
    apt-get update
    apt-get upgrade -y
    apt-get install -y curl
    apt-get clean
    rm -rf /var/lib/apt/lists/*
EOF

RUN python -m pip install --upgrade pip

# Copy uv binary from the source image INTO the builder stage
COPY --from=ghcr.io/astral-sh/uv:0.8.15 /uv /uvx /usr/local/bin/

WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Determine whether to include GPU dependencies
ARG GPU="false"

# Install dependencies into a virtual environment, but NOT the project itself
RUN --mount=type=cache,target=/root/.cache/uv << EOF
    if [ "$GPU" = "true" ]; then
        uv sync --locked --no-install-project --no-editable --no-dev --extra gpu
    else 
        uv sync --locked --no-install-project --no-editable --no-dev
    fi
EOF

# Copy the application source code
COPY . /app

# Install the project itself from the local source
RUN --mount=type=cache,target=/root/.cache/uv << EOF
    if [ "$GPU" = "true" ]; then
        uv sync --locked --no-editable --no-dev --extra gpu
    else
        uv sync --locked --no-editable --no-dev
    fi
EOF

#
# Stage 2: Final
#
FROM python:3.12-slim-trixie AS final

# Update OS and Python/PIP Packages
# Install curl
RUN << EOF
    apt-get update
    apt-get upgrade -y
    apt-get install -y curl
    apt-get clean
    rm -rf /var/lib/apt/lists/*
EOF

RUN python -m pip install --upgrade pip

WORKDIR /app

# Copy the environment from the builder stage
COPY --from=builder /app/.venv /app/.venv

# Set the PATH to include the virtual environment's bin directory
ENV PATH="/app/.venv/bin:$PATH"

# Download NLTK data and models
RUN python -c "import nltk; nltk.download('punkt_tab'); nltk.download('stopwords')"

EXPOSE 8080
CMD ["sh", "-c", "memmachine-sync-profile-schema && memmachine-server"]
