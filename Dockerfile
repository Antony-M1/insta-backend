FROM python:3.10.16-bookworm

WORKDIR /insta-backend

COPY requirements.txt /insta-backend
COPY . /insta-backend

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000:8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]