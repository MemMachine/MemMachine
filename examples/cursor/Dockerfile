# Multi-stage build for Python application
FROM python:3.13-slim AS builder

# Set working directory
WORKDIR /app

# Update OS and install curl
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install uv package manager
COPY --from=ghcr.io/astral-sh/uv:0.8.15 /uv /uvx /usr/local/bin/

# Copy dependency files
COPY pyproject.toml uv.lock README.md ./

# Copy source code
COPY . .

# Install dependencies and project
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-editable --no-dev

# Production stage
FROM python:3.13-slim AS final

# Set working directory
WORKDIR /app

# Update OS and install curl
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python -m pip install --upgrade pip

# Copy the environment from the builder stage
COPY --from=builder /app/.venv /app/.venv

# Copy the application code and necessary files
COPY --from=builder /app/server /app/server
COPY --from=builder /app/pyproject.toml /app/pyproject.toml

# Set the PATH to include the virtual environment's bin directory
ENV PATH="/app/.venv/bin:$PATH"

# Create a non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Create necessary directories and set permissions
RUN mkdir -p /app/data && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port 8001
EXPOSE 8001

# Set default environment variables
ENV MEMORY_BACKEND_URL=http://localhost:8080 \
    CURSOR_MCP_PORT=8001 \
    DATABASE_URL=sqlite:////app/data/bigmemory.db \
    VERIFY_SSL=false \
    DEBUG=false

# Start the application
CMD ["python", "-m", "server.main"]
