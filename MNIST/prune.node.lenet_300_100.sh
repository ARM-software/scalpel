# original training -- 98.48%
python main.py

# stage 0 -- 60 13
python main.py --prune node --stage 0 \
	--pretrained saved_models/LeNet_300_100.best_origin.pth.tar \
	--lr 0.001 --penalty 0.0002 --lr-epochs 30

# stage 1 -- 120 26
python main.py --prune node --stage 1 \
	--pretrained saved_models/LeNet_300_100.prune.node.0.pth.tar \
	--lr 0.001 --penalty 0.0003 --lr-epochs 30

# stage 2 -- 139 36
python main.py --prune node --stage 2 \
	--pretrained saved_models/LeNet_300_100.prune.node.1.pth.tar \
	--lr 0.001 --penalty 0.0010 --lr-epochs 30

# stage 3 retrain -- 98.54%
python main.py --prune node --stage 3 --retrain \
	--pretrained saved_models/LeNet_300_100.prune.node.2.pth.tar \
	--lr 0.1 --lr-epochs 20
