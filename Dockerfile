FROM python:3.11-slim

# Crawl4AI drives a headless Chromium through Playwright. These are the system
# libraries that browser needs; without them the sync fails at launch with a
# missing-shared-object error rather than anything about scraping.
RUN apt-get update && apt-get install -y --no-install-recommends \
        wget gnupg ca-certificates fonts-liberation \
        libnss3 libnspr4 libatk1.0-0 libatk-bridge2.0-0 libcups2 \
        libdrm2 libxkbcommon0 libxcomposite1 libxdamage1 libxfixes3 \
        libxrandr2 libgbm1 libasound2 libpango-1.0-0 libcairo2 \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces runs as a non-root user; everything must be writable by them.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/user/.cache/huggingface

WORKDIR $HOME/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Install the browser binary itself. Separate layer so a code change does not
# re-download ~150MB of Chromium.
RUN python -m playwright install chromium

COPY --chown=user . .

# Bake the models into the image rather than downloading them on first boot.
# Without this the Space takes ~60s to answer its first query while it pulls
# ~500MB from the Hub, which reads as "broken" to anyone trying the demo.
RUN python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('BAAI/bge-base-en-v1.5'); \
CrossEncoder('BAAI/bge-reranker-base')"

EXPOSE 7860

CMD ["uvicorn", "gradio_app:app", "--host", "0.0.0.0", "--port", "7860"]