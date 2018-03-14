COMMON_FLAG='--arch NIN --batch-size 128 --momentum 0.95 --weight-decay 1e-4'
COMMON_FLAG_PRUNE='--beta-initial 0.802 --beta-limit 0.82'

# original training -- 89.67%
python main.py $COMMON_FLAG --lr 0.1 --epochs 320 --lr-epochs 80

# stage 0 -- 5 1 0 2 0 0 8 28 -- 89.92%
python main.py $COMMON_FLAG $COMMON_FLAG_PRUNE --prune node --stage 0 \
	--pretrained saved_models/NIN.best_origin.pth.tar \
	--lr 0.01 --penalty 0.0010 --lr-epochs 40 --epochs 80

# stage 1 -- 35 12 0 3 12 0 44 72 -- 89.93%
python main.py $COMMON_FLAG $COMMON_FLAG_PRUNE --prune node --stage 1 \
	--pretrained saved_models/NIN.prune.node.0.pth.tar \
	--lr 0.01 --penalty 0.0020 --lr-epochs 40 --epochs 80

# stage 2 -- 46 23 1 3 31 7 69 99 -- 89.57%
python main.py $COMMON_FLAG $COMMON_FLAG_PRUNE --prune node --stage 2 \
	--pretrained saved_models/NIN.prune.node.1.pth.tar \
	--lr 0.01 --penalty 0.0030 --lr-epochs 40 --epochs 80

# stage 3 -- 59 39 13 4 43 18 78 117 -- 89.54%
python main.py $COMMON_FLAG $COMMON_FLAG_PRUNE --prune node --stage 3 \
	--pretrained saved_models/NIN.prune.node.2.pth.tar \
	--lr 0.01 --penalty 0.0040 --lr-epochs 40 --epochs 80

# stage 4 -- 68 54 23 4 55 19 87 131 -- 89.58%
python main.py $COMMON_FLAG $COMMON_FLAG_PRUNE --prune node --stage 4 \
	--pretrained saved_models/NIN.prune.node.3.pth.tar \
	--lr 0.01 --penalty 0.0050 --lr-epochs 40 --epochs 80

# stage 5 -- 77 64 29 6 59 25 93 135 -- 89.73%
python main.py $COMMON_FLAG $COMMON_FLAG_PRUNE --prune node --stage 5 \
	--pretrained saved_models/NIN.prune.node.4.pth.tar \
	--lr 0.01 --penalty 0.0060 --lr-epochs 40 --epochs 80

# stage 6 -- 85 79 34 9 69 44 101 138 -- 89.49%
python main.py $COMMON_FLAG $COMMON_FLAG_PRUNE --prune node --stage 6 \
	--pretrained saved_models/NIN.prune.node.5.pth.tar \
	--lr 0.01 --penalty 0.0070 --lr-epochs 40 --epochs 80

# stage 7 retrain -- 89.68%
python main.py $COMMON_FLAG $COMMON_FLAG_PRUNE --prune node --stage 7 --retrain \
	--pretrained saved_models/NIN.prune.node.6.pth.tar \
	--lr 0.001 --lr-epochs 40 --epochs 80
