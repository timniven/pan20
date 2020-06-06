#!/usr/bin/env bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate base
python3 profiling.py $1 $2 --subset 200
