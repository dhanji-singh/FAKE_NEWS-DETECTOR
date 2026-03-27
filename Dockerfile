# syntax=docker/dockerfile:1

FROM python:3.11-slim

# Don't write .pyc files and keep stdout/stderr unbuffered
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .
COPY templates/ templates/
COPY static/ static/

# The model/ directory is mounted at runtime (see docker-compose.yml).
# To bake in pre-trained artifacts instead, uncomment the next line:
# COPY model/ model/

EXPOSE 5000

CMD ["python", "app.py"]
