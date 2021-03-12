#!/bin/sh

#PBS -l select=1:ncpus=8:ngpus=1:mem=24gb:host=rcggpu05

#PBS -l walltime=100:00:00

#PBS -k oe

#PBS -q gpuq

source /etc/profile.d/modules.sh

module load cuda10.1/toolkit/10.1.243
module load torch/1.5.0-python3.6-cudnn7.6
module load scikit-learn/1.19.1-python3.6




cd /home/c3265591/ShipSim/geometric-CFD



python3 create_dataset.py --samples 694 --dilation 1 --seq_length 50 --epochs 1000 --lr 0.0003  --decay_step 500 --exp _len50_conv4_sigmoid_l1loss_0003_2 --resume



exit 0
