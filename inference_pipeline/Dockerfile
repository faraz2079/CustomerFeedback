FROM python:3.12.8
WORKDIR /app
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir awscli
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
ENV OTEL_SERVICE_NAME="feedback-inference-service"
ENV TARGET_ONE_HOST="inference-helper-service"
COPY . .

# Expose the port that FastAPI will run on
EXPOSE 8000

# Command to start the FastAPI app with Uvicorn
ENV WORKERS=4
CMD ["sh", "-c", "gunicorn inference:app -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000 -w $WORKERS"]
