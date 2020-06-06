#!/usr/bin/env bash

python3 -m venv activate pan20env
cd pan20
python3 profile.py $1 $2
