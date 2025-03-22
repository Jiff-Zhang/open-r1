#!/usr/bin/env bash

# ***********************************************
#      Filename: run.sh
#        Author: jiff
#         Email: Jiff_Zh@163.com
#   Description: --
#        Create: 2025-03-12 16:58:38
# Last Modified: Year-month-day
# ***********************************************

set -e

if [[ $# -ne 2 ]]; then
    echo "Usage: bash $0 <data> <outdir>"
    exit
fi

data=$1 && shift
outdir=$1 && shift
model=DeepSeek-R1

# max_new_tokens=32
max_new_tokens=32768

# n_proc=4
# n_proc=16
# n_proc=32
n_proc=64
# n_proc=256
temperature=0.6
top_p=0.95

python pred.py \
    --data $data \
    --model $model \
    --save_dir $outdir \
    --n_proc ${n_proc} \
    --temperature ${temperature} \
    --top_p ${top_p} \
    --max_new_tokens ${max_new_tokens} \

