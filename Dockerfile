FROM python:3.12-slim

WORKDIR /app

# Copy requirements and install before copying code
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application, code, model and configuration
COPY src/ ./src/
COPY app/ ./app/
COPY models/ ./models/
COPY params.yaml .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
