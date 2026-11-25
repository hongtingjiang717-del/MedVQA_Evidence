FROM python:3.12-slim
WORKDIR /app
COPY evidence/requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt
COPY evidence ./evidence
COPY slake_vqa ./slake_vqa
ENV EVIDENCE_HOST=0.0.0.0
EXPOSE 8000
CMD ["python", "-m", "slake_vqa", "demo"]
