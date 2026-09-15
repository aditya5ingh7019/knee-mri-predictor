# Use a lightweight Python base image
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Upgrade pip first to avoid wheel metadata mismatch errors
RUN pip install --no-cache-dir --upgrade pip

# Copy requirements file
COPY requirements.txt .

# Install CPU-only PyTorch using --extra-index-url
RUN pip install --no-cache-dir --default-timeout=1000 torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

# Copy model weights and application code
COPY best_model_2series.pt .
COPY main.py .

# Expose FastAPI port
EXPOSE 8000

# Start Uvicorn server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]