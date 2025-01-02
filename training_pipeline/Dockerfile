# Use Python 3.12 image
FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y gcc python3-dev git && apt-get clean && rm -rf /var/lib/apt/lists/*
# Install AWS CLI for S3 operations
RUN pip install --no-cache-dir awscli
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# Set environment variables for AWS
ENV AWS_ACCESS_KEY_ID=your-aws-access-key-id
ENV AWS_SECRET_ACCESS_KEY=your-aws-secret-access-key
ENV AWS_DEFAULT_REGION=your-region

# Run the training script
CMD ["python3", "trainer.py"]
