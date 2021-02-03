#!/bin/bash
#SBATCH -J 3dfpn_train
#SBATCH -o train_resnet101.log
#SBATCH -N 1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:4
#SBATCH -p p-RTX2080

python main.py --model 3dfpn -b 8 --save-dir fpn3d/retrft960/ --epochs 1000 --config config_cluster --lr 0.0001 --resume 