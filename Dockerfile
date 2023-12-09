FROM python

RUN apt-get update && apt-get

WORKDIR /app

COPY . /app
RUN pip install -r requirements.txt

CMD ["python3" , "app.py"]