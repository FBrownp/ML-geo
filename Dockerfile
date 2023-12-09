FROM python: 3.11.4

RUN apt-get update && apt-get

WORKDIR /app

COPY . /app
RUN pip install -r requirements.txt

CMD ["python" , "app.py"]