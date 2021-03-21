FROM python:3.9.1-buster

WORKDIR /root/

RUN apt-get update ##[edited]
RUN apt-get install ffmpeg libsm6 libxext6  -y

ADD requirements.txt /root/requirements.txt

ADD calculator.py /root/calculator.py
ADD processor.py /root/processor.py
ADD d2coding.ttc /root/d2coding.ttc
ADD app.py /root/app.py
ADD templates /root/templates

RUN pip install -r requirements.txt

EXPOSE 8080

CMD python app.py
