python main.py --model 3dfpn -b 1 --resume results/fpn3d/retrft960/170.ckpt --test 1 --save-dir fpn3d/retrft960/ --config config_training0
mkdir results/fpn3d/retrft960/val170
mv results/fpn3d/retrft960/bbox/* results/fpn3d/retrft960/val170
