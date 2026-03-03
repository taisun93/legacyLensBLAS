# Deploying LegacyLens on Render

## Prerequisites

- Git repo containing `legacylens/` (or use `legacylens/` as repo root)
- VOYAGE_API_KEY and ANTHROPIC_API_KEY (add in Render dashboard)
- `chroma_db/` committed to the repo (run `python ingest.py` locally once, then commit)

## Deploy via Blueprint

1. In [Render Dashboard](https://dashboard.render.com), click **New** → **Blueprint**
2. Connect your Git repo
3. Render will read `render.yaml` and create a Web Service
4. Add environment variables in the service:
   - `VOYAGE_API_KEY`
   - `ANTHROPIC_API_KEY`

## Deploy via Manual Setup

1. **New** → **Web Service**
2. Connect your repo
3. **Root Directory:** `legacylens` (if it's a subfolder)
4. **Build Command:** `pip install -r requirements.txt`
5. **Start Command:** `./start.sh` (or `sh -c 'uvicorn api:app --host 0.0.0.0 --port $PORT'`)
6. **Environment:** Add `VOYAGE_API_KEY`, `ANTHROPIC_API_KEY`

## Before First Deploy

Run `python ingest.py` locally once to create `chroma_db/`, then commit and push it. The embeddings are included in the repo so the build skips ingestion (faster builds, no Voyage API calls during deploy).
