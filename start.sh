#!/usr/bin/env bash

apt-get install python-dev python-pip

pip3 install -U virtualenv

virtualenv -p python3 env-dlkeras
source env-dlkeras/bin/activate
pip install --upgrade -r requirements.txt

python python_script.py
