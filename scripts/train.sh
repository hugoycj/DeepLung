#python main.py --model dpn3d26 -b 2  --save-dir dpn3d26/retrft960/ --epochs 1000 --config config_training0
#python main.py --model 3dfpn -b 2 --save-dir fpn3d/retrft960/ --epoch
python main.py --model 3dfpn -b 4 --save-dir fpn3d/retrft960/ --epochs 1000 --config config_cluster --lr 0.0001 --save-freq 10
