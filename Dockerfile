# Same image runs on Render (free tier) today and Cloud Run later: it listens on $PORT,
# which both platforms set. No secrets are baked in; all config arrives as env vars.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080

RUN useradd --create-home --uid 1000 app
WORKDIR /home/app/srv

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY --chown=app:app sbrag/ sbrag/
USER app

EXPOSE 8080
# Shell form so $PORT expands at runtime.
CMD uvicorn sbrag.api:app --host 0.0.0.0 --port ${PORT} --proxy-headers --no-access-log
