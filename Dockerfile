FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN pip install --no-cache-dir -e .

EXPOSE 8000

# Serve the online feature API.
CMD ["uvicorn", "featurestore.api:app", "--host", "0.0.0.0", "--port", "8000"]
