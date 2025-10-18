# Sapheneia TimesFM - Hugging Face Spaces Dockerfile
# Optimized for deployment on Hugging Face Spaces with Docker SDK

# Use Python 3.11 slim image for smaller size
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security (HF Spaces best practice)
RUN useradd -m -u 1000 sapheneia

# Copy requirements first for better layer caching
COPY --chown=sapheneia:sapheneia webapp/requirements.txt /app/requirements.txt
COPY --chown=sapheneia:sapheneia pyproject.toml /app/pyproject.toml

# Install Python dependencies
# Use PyTorch backend for best compatibility across hardware
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir \
    timesfm[torch]>=1.3.0 \
    jax>=0.7.0 \
    jaxlib>=0.7.0 \
    plotly>=5.0.0

# Copy application code
COPY --chown=sapheneia:sapheneia src/ /app/src/
COPY --chown=sapheneia:sapheneia webapp/ /app/webapp/
COPY --chown=sapheneia:sapheneia data/ /app/data/

# Create necessary directories with proper permissions
# /app/.cache is used for model downloads (writable by sapheneia user)
RUN mkdir -p /app/webapp/uploads /app/webapp/results /app/logs /app/.cache && \
    chown -R sapheneia:sapheneia /app

# Set environment variables
# Use /app/.cache for model caching (guaranteed writable)
# On HF Spaces with persistent storage, you can override to /data/.cache
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/.cache \
    TRANSFORMERS_CACHE=/app/.cache \
    HF_HUB_CACHE=/app/.cache \
    FLASK_APP=webapp/app.py \
    PORT=7860

# Switch to non-root user
USER sapheneia

# Expose port 7860 (required by Hugging Face Spaces)
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Change to webapp directory and run the application
WORKDIR /app/webapp
CMD ["python", "app.py"]
