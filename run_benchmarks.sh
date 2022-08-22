#!/bin/bash

python benchmark.py --model_name deeplabv3plus_mobileones1
python benchmark.py --model_name deeplabv3plus_mobileones1 --batch_size 8
python benchmark.py --model_name deeplabv3plus_mobileones1 --batch_size 16
python benchmark.py --model_name deeplabv3plus_mobileones1 --batch_size 32

python benchmark.py --model_name deeplabv3plus_mobilenetv2 --batch_size 8
python benchmark.py --model_name deeplabv3plus_mobilenetv2 --batch_size 16
python benchmark.py --model_name deeplabv3plus_mobilenetv2 --batch_size 32