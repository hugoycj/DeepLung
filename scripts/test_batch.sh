#!/bin/bash
set -e

for (( i=10; i<=90; i+=10))
do
    echo "process $i epoch"
    python main.py --model 3dfpn -b 1 --resume results/fpn3d/retrft960/0$i.ckpt --test 1 --save-dir fpn3d/retrft960/ --config config_training0

    if [ ! -d "results/fpn3d/retrft960/val$i/" ]; then
        mkdir results/fpn3d/retrft960/val$i/
    fi
    mv results/fpn3d/retrft960/bbox/*.npy results/fpn3d/retrft960/val$i/
done