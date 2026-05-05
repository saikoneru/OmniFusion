FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    build-essential \
    python3-dev \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*


COPY . .
RUN pip install --no-cache-dir -r requirements.txt

ARG CACHE_BUSTER=1

# Default port (can be overridden by docker-compose)
ENV FLASK_PORT=8001
ENV PYTHONUNBUFFERED=1

# Expose that port dynamically
EXPOSE ${WORKER_PORT}
# Run your app using the environment variable
CMD ["python", "-u", "app.py"]
