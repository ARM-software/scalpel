COMMON_FLAGS='--arch LeNet_5'

# original training -- 99.34%
python main.py $COMMON_FLAGS

# stage 0 -- 3 29 0
python main.py $COMMON_FLAGS --prune node --prune-target conv --stage 0 \
	--pretrained saved_models/LeNet_5.best_origin.pth.tar \
	--lr 0.001 --penalty 0.001 --lr-epochs 30

# stage 1 -- 6 32 0
python main.py $COMMON_FLAGS --prune node --prune-target conv --stage 1 \
	--pretrained saved_models/LeNet_5.prune.node.0.pth.tar \
	--lr 0.001 --penalty 0.004 --lr-epochs 30

# stage 2 -- 8 30 0
python main.py $COMMON_FLAGS --prune node --prune-target conv --stage 2 \
	--pretrained saved_models/LeNet_5.prune.node.1.pth.tar \
	--lr 0.001 --penalty 0.006 --lr-epochs 30

# stage 3 -- 11 32 0
python main.py $COMMON_FLAGS --prune node --prune-target conv --stage 3 \
	--pretrained saved_models/LeNet_5.prune.node.2.pth.tar \
	--lr 0.001 --penalty 0.009 --lr-epochs 30

# stage 4 -- 11 32 435
python main.py $COMMON_FLAGS --prune node --prune-target ip --stage 4 \
	--pretrained saved_models/LeNet_5.prune.node.3.pth.tar \
	--lr 0.001 --penalty 0.0015 --lr-epochs 30

# stage 5 -- 99.34%
python main.py $COMMON_FLAGS --prune node --retrain --stage 5 \
	--pretrained saved_models/LeNet_5.prune.node.4.pth.tar \
	--lr 0.001 --lr-epochs 30
