#!/bin/bash
#SBATCH -J deeplung_test
#SBATCH -o test.out
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
set -e

module load anaconda3
module load cuda10.1/blas/10.1.243 
source /home/yechongjie/.bashrc

conda activate py27
cd detector

python2 main.py --model res18 -b 32 --resume /data_set_medical/lyshi/LUNA16/Models_for_DeepLung/res18/072.ckpt --test 1 --save-dir /data_set_medical/lyshi/LUNA16/val_results --config config_training