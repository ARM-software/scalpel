# original training -- 81.38%
python main.py

# stage 0 -- 0 10 16
python main.py --prune node --stage 0 \
	--pretrained saved_models/ConvNet.best_origin.pth.tar \
	--lr 0.0001 --penalty 0.03 --lr-epochs 30 --epochs 60

# stage 1 -- 0 10 18
python main.py --prune node --stage 1 \
 	--pretrained saved_models/ConvNet.prune.node.0.pth.tar \
 	--lr 0.0001 --penalty 0.04 --lr-epochs 30 --epochs 60

# stage 2 -- 1 10 20
python main.py --prune node --stage 2 \
 	--pretrained saved_models/ConvNet.prune.node.1.pth.tar \
 	--lr 0.0001 --penalty 0.05 --lr-epochs 30 --epochs 60
 
# stage 3 -- 4 10 24
python main.py --prune node --stage 3 \
 	--pretrained saved_models/ConvNet.prune.node.2.pth.tar \
 	--lr 0.0001 --penalty 0.06 --lr-epochs 30 --epochs 60
 
# retrain -- 81.52%
python main.py --prune node --stage 4 --retrain \
	--pretrained saved_models/ConvNet.prune.node.3.pth.tar \
	--lr 0.1 --lr-epochs 10 --epochs 40
