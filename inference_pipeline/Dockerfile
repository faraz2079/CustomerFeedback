# Use Python 3.12 slim as the base image
FROM python:3.12-slim
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
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# Expose the port that FastAPI will run on
EXPOSE 8000

# Command to start the FastAPI app with Uvicorn
CMD ["uvicorn", "inference:app", "--host", "0.0.0.0", "--port", "8000"]
