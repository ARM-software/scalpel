# original training -- 98.48%
python main.py

# 60.6% pruned
python main.py --prune simd --stage 0 --width 8\
	--pretrained saved_models/LeNet_300_100.best_origin.pth.tar \
	--lr 0.01 --lr-epochs 20 --threshold 0.04

# 72.6% pruned
python main.py --prune simd --stage 1 --width 8\
	--pretrained saved_models/LeNet_300_100.prune.simd.0.pth.tar \
	--lr 0.01 --lr-epochs 20 --threshold 0.05

# 82.4% pruned
python main.py --prune simd --stage 2 --width 8\
	--pretrained saved_models/LeNet_300_100.prune.simd.1.pth.tar \
	--lr 0.01 --lr-epochs 20 --threshold 0.06

# 88.7% pruned
python main.py --prune simd --stage 3 --width 8\
	--pretrained saved_models/LeNet_300_100.prune.simd.2.pth.tar \
	--lr 0.01 --lr-epochs 20 --threshold 0.07

# 92.0% pruned
python main.py --prune simd --stage 4 --width 8\
	--pretrained saved_models/LeNet_300_100.prune.simd.3.pth.tar \
	--lr 0.01 --lr-epochs 20 --threshold 0.08
