FROM python:3.12-slim

WORKDIR /app

RUN echo "Installing system dependencies"
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN echo "Installing Server dependencies"
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN echo "Starting Server"
CMD ["python3", "main"]
