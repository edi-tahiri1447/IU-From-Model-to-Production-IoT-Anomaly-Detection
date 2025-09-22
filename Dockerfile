# Use official Python image
FROM python:3.11-slim

# Env settings
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Workdir
WORKDIR /app

# System dependencies (needed for psycopg2 & numpy/pandas/scikit-learn if used)
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose Flask/Gunicorn port
EXPOSE 8000

# Start the app
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8000"]