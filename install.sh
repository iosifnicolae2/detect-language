#!/usr/bin/env bash

pip3 install -U virtualenv

pip install --upgrade pip

virtualenv -p python3 env-dlkeras
source env-dlkeras/bin/activate
pip install --upgrade -r requirements.txt
