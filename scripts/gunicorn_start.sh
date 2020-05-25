#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate tf_gpu

cd web_service && gunicorn -b 127.0.0.1:5000 --reload fer_service:app
