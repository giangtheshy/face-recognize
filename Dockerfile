FROM python:3.11.9-slim-bullseye

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "master:app", "--host", "0.0.0.0", "--port", "8000"]