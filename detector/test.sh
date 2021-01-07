source ~/.bashrc
conda activate py2.7
python2 main.py --model res18 -b 1 --resume results/res18/retrft96/078.ckpt --test 1 --save-dir res18/retrft96/ --config config_training0

mv results/res18/retrft96/bbox/*.npy results/res18/retrft96/val/