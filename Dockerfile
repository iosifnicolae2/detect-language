FROM tensorflow/tensorflow:latest-gpu

WORKDIR /usr/src/app

COPY . .

RUN bash install.sh
