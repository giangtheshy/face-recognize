FROM python:3.11.9-slim-bullseye

# Install dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0
    
WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY . .

# EXPOSE 8000

# CMD ["uvicorn", "master:app", "--host", "0.0.0.0", "--port", "8000"]