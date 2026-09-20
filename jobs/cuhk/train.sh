#!/bin/bash

config_name="cuhk_dicl"
config_path="../../configs/dicl/${config_name}.py" 
unset LD_LIBRARY_PATH
CUDA_VISIBLE_DEVICES=5 python -u ../../tools/train.py ${config_path} >train_log.txt 2>&1 
# CUDA_VISIBLE_DEVICES=1 python -u ../../tools/train.py ../../configs/dicl/cuhk_dicl.py >train_cuhk_dicl.txt 2>&1