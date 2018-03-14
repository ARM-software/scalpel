COMMON_FLAGS='--arch LeNet_5'
SIMD_WIDTH=8

# original training -- 99.34%
python main.py $COMMON_FLAGS

# 83.4% pruned
python main.py $COMMON_FLAGS --prune simd --stage 0 --width $SIMD_WIDTH\
	--pretrained saved_models/LeNet_5.best_origin.pth.tar \
	--lr 0.1 --lr-epochs 15 --threshold 0.04

# 92.1% pruned
python main.py $COMMON_FLAGS --prune simd --stage 1 --width $SIMD_WIDTH\
	--pretrained saved_models/LeNet_5.prune.simd.0.pth.tar \
	--lr 0.01 --lr-epochs 20 --threshold 0.05

# 93.6% pruned
python main.py $COMMON_FLAGS --prune simd --stage 2 --width $SIMD_WIDTH\
	--pretrained saved_models/LeNet_5.prune.simd.1.pth.tar \
	--lr 0.01 --lr-epochs 20 --threshold 0.06

# 95.9% pruned
python main.py $COMMON_FLAGS --prune simd --stage 3 --width $SIMD_WIDTH\
	--pretrained saved_models/LeNet_5.prune.simd.2.pth.tar \
	--lr 0.01 --lr-epochs 20 --threshold 0.075

# 96.8% pruned
python main.py $COMMON_FLAGS --prune simd --stage 4 --width $SIMD_WIDTH\
	--pretrained saved_models/LeNet_5.prune.simd.3.pth.tar \
	--lr 0.01 --lr-epochs 20 --threshold 0.080
