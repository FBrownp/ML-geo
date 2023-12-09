FROM python:3.11


WORKDIR /app

COPY . /app
RUN pip install --no-cache-dir --upgrade pip \
  && pip install --no-cache-dir -r requirements.txt

# CMD ["python3" , "app.py"]

CMD ["uvicorn", "app.app:app", "--reload", "--host", "0.0.0.0", "--port", "80"]
