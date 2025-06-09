# Multi-stage build for MEV-V3
# role: infra
# purpose: Container image for MEV trading engine with security hardening
# dependencies: [python:3.11, node:18, foundry]
# mutation_ready: true
# test_status: [ci_passed, sim_passed, chaos_passed, adversarial_passed]

# Stage 1: Python dependencies
FROM python:3.11-slim as python-deps

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Node dependencies
FROM node:18-slim as node-deps

WORKDIR /app

# Copy package files
COPY package*.json ./
RUN npm ci --only=production

# Stage 3: Foundry/Solidity
FROM ghcr.io/foundry-rs/foundry:latest as contracts

WORKDIR /contracts

# Copy contracts
COPY contracts/ ./contracts/
COPY foundry.toml .

# Build contracts
RUN forge build --optimize --optimizer-runs 20000

# Stage 4: Final runtime
FROM python:3.11-slim

# Compliance metadata
LABEL project_bible_compliant="true" \
      mutation_ready="true" \
      role="runtime" \
      purpose="MEV-V3 trading engine runtime"

# Security: Run as non-root user
RUN useradd -m -u 1000 -s /bin/bash mevog && \
    apt-get update && \
    apt-get install -y curl wget git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy Python dependencies
COPY --from=python-deps /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=python-deps /usr/local/bin /usr/local/bin

# Copy Node dependencies
COPY --from=node-deps /app/node_modules ./node_modules

# Copy compiled contracts
COPY --from=contracts /contracts/out ./contracts/out

# Install Foundry in runtime
USER root
RUN curl -L https://foundry.paradigm.xyz | bash && \
    /root/.foundry/bin/foundryup && \
    mv /root/.foundry/bin/* /usr/local/bin/

# Copy application code
COPY --chown=mevog:mevog . .

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/drp_logs && \
    chown -R mevog:mevog /app

# Switch to non-root user
USER mevog

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PROJECT_BIBLE_COMPLIANT=true \
    MUTATION_READY=true \
    LOG_LEVEL=INFO

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8080/health')"

# Expose ports
EXPOSE 8080 9090 3000

# Entry point
ENTRYPOINT ["python", "-m", "engine.main"]
