# Use Python 3.11 for better package compatibility
FROM python:3.11.9-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies for OpenCV and other packages
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    libgtk-3-0 \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file first for better Docker layer caching
COPY requirements.txt .

# Upgrade pip and setuptools
RUN pip install --upgrade pip setuptools wheel

# Install PyTorch CPU version first
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install other Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code
COPY . .

# Create a non-root user for security
RUN useradd --create-home --shell /bin/bash app && chown -R app:app /app
USER app

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Run the app when the container launches
CMD ["python", "main.py"]
