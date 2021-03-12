#!/bin/sh

#PBS -l select=1:ncpus=8:ngpus=1:mem=24gb:host=rcggpu05

#PBS -l walltime=100:00:00

#PBS -k oe

#PBS -q gpuq

source /etc/profile.d/modules.sh

module load cuda10.1/toolkit/10.1.243
module load tensorflow/1.15.0-python3.6-cudnn7.6
module load torch/1.5.0-python3.6-cudnn7.6
module load scikit-learn/1.19.1-python3.6


 

cd /home/c3265591/ShipSim/RandLA-Net-pytorch



python3 train.py --samples 100 --save_freq 10 --decimation 2 --create_dataset --dataset ./../geometric-CFD/data/

 

exit 0
