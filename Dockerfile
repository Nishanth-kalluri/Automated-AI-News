# Same environment as the GitHub Actions run, for any container host (Cloud Run Jobs, a VPS, Modal).
FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY shorts ./shorts
RUN pip install --no-cache-dir ".[all]"

# Mount or sync /app/state between runs so already-aired stories aren't repeated.
ENTRYPOINT ["python", "-m", "shorts"]
CMD ["run", "--upload"]
