FROM python:3.12-alpine

# Install system dependencies including Rust
RUN apk add --no-cache \
    bash \
    git \
    gcc \
    musl-dev \
    linux-headers \
    build-base \
    rust \
    cargo

WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ENVIRONMENT=production

# Copy pyproject.toml first for better Docker layer caching
COPY pyproject.toml .
COPY app ./app

# Install dependencies from pyproject.toml
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .


# Create logs directory and non-root user
RUN mkdir -p /app/logs && \
    adduser -D user && \
    chown -R user:user /app
USER user


CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9002","--workers", "4"]