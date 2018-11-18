#!/usr/bin/env bash

apt update
apt install python-dev
apt install python3-pip
pip3 --version

pip3 install -U virtualenv

virtualenv -p python3 env-dlkeras
source env-dlkeras/bin/activate
pip install --upgrade -r requirements.txt

python python_script.py
