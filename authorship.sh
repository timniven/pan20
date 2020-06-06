#!/usr/bin/env bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate base
python3 authorship.py $1 $2
