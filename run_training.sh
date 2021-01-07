#!/bin/bash
#SBATCH -J deeplung_test
#SBATCH -o train.out
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
set -e

module load anaconda3
module load cuda10.1/blas/10.1.243 
source /home/yechongjie/.bashrc
#conda init bash

conda activate py27
# source activate py27

# python prepare.py
cd detector
maxeps=150
f=9
python2 main.py --model res18 -b 64 --resume 064.ckpt --save-dir res18/retrft96$f/ --epochs $maxeps --config config_training
for (( i=1; i<=$maxeps; i+=1)) 
do
    echo "process $i epoch"
	
	if [ $i -lt 10 ]; then
	    python2 main.py --model res18 -b 32 --resume results/res18/retrft96$f/00$i.ckpt --test 1 --save-dir res18/retrft96$f/ --config config_training
	elif [ $i -lt 100 ]; then 
	    python2 main.py --model res18 -b 32 --resume results/res18/retrft96$f/0$i.ckpt --test 1 --save-dir res18/retrft96$f/ --config config_training
	elif [ $i -lt 1000 ]; then
	    python2 main.py --model res18 -b 32 --resume results/res18/retrft96$f/$i.ckpt --test 1 --save-dir res18/retrft96$f/ --config config_training
	else
	    echo "Unhandled case"
    fi

    if [ ! -d "results/res18/retrft96$f/val$i/" ]; then
        mkdir results/res18/retrft96$f/val$i/
    fi
    mv results/res18/retrft96$f/bbox/*.npy results/res18/retrft96$f/val$i/
done 