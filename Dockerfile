FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
# Install AWS CLI dependencies for DVC, which may resolve some low-level library issues
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .

# *** FIX: Removing --use-deprecated=legacy-resolver for a cleaner, stable install ***
# This forces pip to resolve dependencies normally, which is often cleaner.
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy application code including .dvc directory
COPY . .

# Create necessary directories
RUN mkdir -p models data/raw data/processed

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Run the application
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]