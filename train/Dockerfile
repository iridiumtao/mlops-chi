FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    prefect \
    mlflow==2.20.2 \
    fastapi \
    uvicorn \
    pytest

# Copy application files
COPY flow.py /app/flow.py
COPY food11.pth /app/food11.pth
COPY bad_model.pth /app/bad_model.pth
COPY oversized_model.pth /app/oversized_model.pth
COPY oversized_model.py /app/oversized_model.py
COPY tests/ /app/tests/

# Expose port for FastAPI
EXPOSE 8000

# Start FastAPI server
ENTRYPOINT ["python", "flow.py"]
