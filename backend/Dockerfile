FROM python:3.11-slim as base
WORKDIR /app
COPY backend/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

FROM base as runtime
WORKDIR /app
COPY backend/app /app/app
COPY example /app/example
COPY docs /app/docs
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
