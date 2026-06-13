# Green Computing - Carbon-Aware HRL for Data Center Energy Optimization
# Multi-stage build for optimized production image

# Stage 1: Builder
FROM python:3.10-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Production
FROM python:3.10-slim as production

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY src/ src/
COPY main.py .
COPY pyproject.toml .

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser
USER appuser

# Default command
ENTRYPOINT ["python", "main.py"]
CMD ["demo"]

# Labels
LABEL maintainer="Green Computing Team"
LABEL description="Carbon-Aware Hierarchical RL for Data Center Energy Optimization"
LABEL version="1.0.0"
