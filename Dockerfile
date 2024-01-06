FROM nvcr.io/nvidia/pytorch:23.12-py3

WORKDIR /e5-mistral-7b-instruct/

COPY requirements.txt /tmp/requirements.txt

RUN pip install -U pip

RUN pip install -r /tmp/requirements.txt

