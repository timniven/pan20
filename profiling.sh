#!/usr/bin/env bash

conda activate base
python3 profiling.py $1 $2 --subset 200
