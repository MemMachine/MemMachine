#
# Stage 1: Builder
#
FROM python:3.12-slim-trixie AS builder

# Copy uv binary from the source image INTO the builder stage
COPY --from=ghcr.io/astral-sh/uv:0.8.15 /uv /uvx /usr/local/bin/

WORKDIR /app

# Copy dependency files only
COPY pyproject.toml uv.lock ./

# Install dependencies into a virtual environment, but NOT the project itself
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project

#
# Stage 2: Final
#
FROM python:3.12-slim-trixie AS final

WORKDIR /app

# Copy the virtual environment from the builder stage
COPY --from=builder /app/.venv /app/.venv

# Copy the uv binary from the builder stage
# The path is now /usr/local/bin/uv and /usr/local/bin/uvx
COPY --from=builder /usr/local/bin/uv /usr/local/bin/uvx /usr/local/bin/

# Set the PATH to include the virtual environment's bin directory
ENV PATH="/app/.venv/bin:$PATH"

# Copy the rest of the application source code
COPY . /app

# Now, install the project itself from the local source
RUN uv sync --locked

# Download NLTK data and models
RUN python -c "import nltk; nltk.download('punkt_tab'); nltk.download('stopwords')" && \
    python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2')" && \
    python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/qnli-electra-base')"

EXPOSE ${PORT:-8080}
CMD ["sh", "-c", "memmachine-sync-profile-schema && memmachine-server"]