COMMON_FLAG_PRUNE='--beta-initial 0.802 --beta-limit 0.81'


# original training -- 57.4% 80.3%
python main.py

# stage 0 - 5 1 11 29 0 0 0
python main.py $COMMON_FLAG_PRUNE --prune node --prune-target conv --stage 0 \
	--pretrained saved_models/AlexNet.best_original.pth.tar \
	--lr 0.001 --penalty 0.0030 --lr-epochs 10 --epochs 20 --print-freq-mask 100

# stage 1 - 7 3 52 63 0 0 0
python main.py $COMMON_FLAG_PRUNE --prune node --prune-target conv --stage 1 \
	--pretrained saved_models/AlexNet.prune.node.0.pth.tar \
	--lr 0.001 --penalty 0.0040 --lr-epochs 10 --epochs 20 --print-freq-mask 100 > stage.1.log

# stage 2 - 9 12 92 103 0 0 0
python main.py $COMMON_FLAG_PRUNE --prune node --prune-target conv --stage 2 \
	--pretrained saved_models/AlexNet.prune.node.1.pth.tar \
	--lr 0.001 --penalty 0.0050 --lr-epochs 10 --epochs 20 --print-freq-mask 100 > stage.2.log

# stage 3 - 13 19 122 130
python main.py $COMMON_FLAG_PRUNE --prune node --prune-target conv --stage 3 \
	--pretrained saved_models/AlexNet.prune.node.2.pth.tar \
	--lr 0.001 --penalty 0.0055 --lr-epochs 10 --epochs 20 --print-freq-mask 100 > stage.3.log

# stage 4 - 13 31 151 146 3 0 0
python main.py $COMMON_FLAG_PRUNE --prune node --prune-target conv --stage 4 \
	--pretrained saved_models/AlexNet.prune.node.3.pth.tar \
	--lr 0.001 --penalty 0.0060 --lr-epochs 10 --epochs 20 --print-freq-mask 100 > stage.4.log

# stage 5
python main.py $COMMON_FLAG_PRUNE --prune node --prune-target conv --stage 5 --retrain\
	--pretrained saved_models/AlexNet.prune.node.4.pth.tar \
	--lr 0.01 --penalty 0.0060 --lr-epochs 15 --epochs 45 --print-freq-mask 100 > stage.5.log

# stage 6 - 13 31 151 146 3 458 332 
python main.py $COMMON_FLAG_PRUNE --prune node --prune-target ip --stage 6 \
	--pretrained saved_models/AlexNet.prune.node.5.pth.tar \
	--lr 0.001 --penalty 0.0005 --lr-epochs 10 --epochs 20 --print-freq-mask 100 > stage.6.log

# stage 7 - 13 31 151 146 3 1097 1068
python main.py $COMMON_FLAG_PRUNE --prune node --prune-target ip --stage 7 \
	--pretrained saved_models/AlexNet.prune.node.6.pth.tar \
	--lr 0.001 --penalty 0.0010 --lr-epochs 10 --epochs 20 --print-freq-mask 100 > stage.7.log

# stage 8 - 57.4% 80.5%
python main.py $COMMON_FLAG_PRUNE --prune node --prune-target ip --stage 8 --retrain\
	--pretrained saved_models/AlexNet.prune.node.7.pth.tar \
	--lr 0.01 --penalty 0.0010 --lr-epochs 25 --epochs 100 --print-freq-mask 100 > stage.8.log
