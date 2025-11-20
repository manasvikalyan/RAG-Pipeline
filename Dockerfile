# Use Python base image
FROM python:3.10

# Install curl for health checks
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# Create user (required for HF Spaces)
RUN useradd -m user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /home/user/app

# Copy all project files
COPY --chown=user . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose ports
EXPOSE 8000
EXPOSE 8501

# Start FastAPI in background, wait for health, then start Streamlit
CMD uvicorn api:app --host 0.0.0.0 --port 8000 & \
    for i in $(seq 1 30); do \
      STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health || true); \
      if [ "$STATUS" = "200" ]; then \
        echo "API health check passed"; \
        break; \
      fi; \
      echo "Waiting for API to be healthy... attempt $i"; \
      sleep 1; \
    done; \
    if [ "$STATUS" != "200" ]; then \
      echo "API failed health check, not starting Streamlit"; \
      exit 1; \
    fi; \
    streamlit run app_api.py --server.port 8501 --server.address 0.0.0.0
