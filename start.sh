#!/usr/bin/env bash

apt update
apt install -y python-dev
apt install -y python3-pip
pip3 --version

pip3 install -U virtualenv

virtualenv -p python3 env-dlkeras
source env-dlkeras/bin/activate
pip install --upgrade -r requirements.txt

python3 python_script.py
