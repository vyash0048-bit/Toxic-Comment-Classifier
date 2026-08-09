# ── Stage 1: Build ──────────────────────────────────────────────────
FROM python:3.12-slim AS builder

# Install system dependencies needed for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy only the deploy requirements to leverage Docker layer caching
COPY requirements-deploy.txt ./

# Install dependencies into a clean prefix
RUN pip install --no-cache-dir --prefix=/install -r requirements-deploy.txt

# ── Stage 2: Runtime ────────────────────────────────────────────────
FROM python:3.12-slim

# Create a non-root user
RUN useradd -m -u 1000 user

# Copy installed Python packages from the builder stage
COPY --from=builder /install /usr/local

# Set environment variables
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR /home/user/app

# Copy application code (respects .dockerignore)
COPY --chown=user:user . .

USER user

# Expose port 5000 for the Flask web server
EXPOSE 5000

# Start the Flask app
ENV PORT=5000
CMD ["python", "flask_app.py"]
