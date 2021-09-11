# FROM ubuntu:latest
FROM jjanzic/docker-python3-opencv:latest

# Install some basic utilities and python
RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 \
  # && apt-get install -y python3-pip python3-dev ffmpeg libsm6 libxext6 \
  # && cd /usr/local/bin \
  # && ln -s /usr/bin/python3 python \
  && pip3 install --upgrade pip

RUN pip3 install numpy simpleitk nibabel pydicom scipy opencv-python

COPY ./* ./

RUN groupadd -r myuser -g 433 && \
    useradd -u 431 -r -g myuser -s /sbin/nologin -c "Docker image user" myuser

# Uncomment this line if building for deployment to grand-challenge.org
# USER myuser

CMD python3 ./cli.py
