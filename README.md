# RailSense

Minimal Streamlit chatbot that routes to:
- **Task 1**: ticket search (National Rail SOAP)
- **Task 2**: delay prediction (saved Random Forest model)
- **Task 3**: general rail chat (local Ollama model)

## Quick run

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
docker compose up -d pgvector
docker compose exec -T pgvector psql -U railsense -d railsense < db/schema.sql
streamlit run ui/streamlit_app.py
```

## Required `.env`

```env
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=batiai/gemma4-e2b:q4

OJP_BASIC_AUTH_USER=...
OJP_BASIC_AUTH_PASSWORD=...
OJP_SOAP_ENDPOINT=https://ojp.nationalrail.co.uk/webservices
DATABASE_URL=postgresql://railsense:railsense@localhost:5432/railsense

RAILSENSE_DEBUG=false
```

Set `RAILSENSE_DEBUG=true` to see detailed request/debug info in the UI.
