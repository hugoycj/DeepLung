python main.py --model 3dfpn -b 1 --resume results/fpn3d/retrft960/300.ckpt --test 1 --save-dir fpn3d/retrft960/ --config config_training0
mkdir fpn3d/retrft960/val300
mv results/fpn3d/retrft960/bbox/* results/fpn3d/retrft960/val300
