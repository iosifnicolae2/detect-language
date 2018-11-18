#!/usr/bin/env bash

virtualenv -p python3 env-dlkeras
source env-dlkeras/bin/activate
pip install -r requirements.txt

python3 python_script.py
