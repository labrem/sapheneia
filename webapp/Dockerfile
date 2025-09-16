# Sapheneia TimesFM Web Application Dockerfile
# Multi-stage build for production deployment

# Stage 1: Base Python environment
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Stage 2: Dependencies installation
FROM base as deps

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Stage 3: Application
FROM deps as app

# Copy application code
COPY . .

# Copy src modules from parent directory
COPY ../src ./src

# Create necessary directories
RUN mkdir -p uploads results logs

# Set permissions
RUN chmod +x . && \
    chown -R nobody:nogroup /app

# Switch to non-root user for security
USER nobody

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1

# Default command for production
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "2", "--timeout", "120", "--worker-class", "sync", "app:app"]