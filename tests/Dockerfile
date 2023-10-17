FROM python:3.10-slim as builder
WORKDIR /app
COPY tests .
RUN pip install -r requirements.txt

CMD ["pytest", "api_test.py"]
