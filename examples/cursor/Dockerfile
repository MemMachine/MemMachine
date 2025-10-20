# Use Python 3.13 slim image as base
FROM python:3.13-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for faster Python package management
RUN pip install uv

# Copy dependency files first for better layer caching
COPY pyproject.toml uv.lock ./

# Resolve --allow flag issue by using uv sync with explicit environment setup
# Set UV_NO_UPDATE to avoid update checks that might trigger --allow flag issues
ENV UV_NO_UPDATE=1
RUN uv sync --frozen --no-dev

# Copy the application code
COPY . .

# Create a non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Create necessary directories and set permissions
RUN mkdir -p /app/data && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose the port the app runs on
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/api/health || exit 1

# Set default environment variables
ENV MEMORY_BACKEND_URL=http://localhost:8080 \
    CURSOR_MCP_PORT=8001 \
    DATABASE_URL=sqlite:////app/data/bigmemory.db \
    VERIFY_SSL=false \
    DEBUG=false

# Command to run the application
CMD ["uv", "run", "python", "-m", "server.main"]
