source ~/.bashrc
conda activate py2.7
python2 main.py --model res18 -b 2 --resume 064.ckpt --save-dir res18/retrft96/ --epochs 150 --config config_training0
