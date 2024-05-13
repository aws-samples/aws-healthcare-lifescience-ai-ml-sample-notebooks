FROM --platform=linux/amd64 python:3.12

WORKDIR /deployment

COPY ./requirements.txt /deployment/requirements.txt
COPY ./images /deployment/images

#install Bedrock client
RUN cd /deployment/ &&  pip install --upgrade pip

RUN pip install --no-cache-dir --upgrade -r /deployment/requirements.txt

COPY app.py /deployment/

EXPOSE 8080

CMD ["python", "app.py"]

