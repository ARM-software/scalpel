# Scalpel

This is a PyTorch implementation of the [Scalpel](http://www-personal.umich.edu/~jiecaoyu/papers/jiecaoyu-isca17.pdf). Node pruning for five benchmark networks and SIMD-aware weight pruning for LeNet-300-100 and LeNet-5 is included.

# Node Pruning
Here is the results for node pruning:
<table>
  <tr>
    <th rowspan="2">Network</th>
    <th rowspan="2">Dataset</th>
    <th rowspan="2">Structure After Pruning</th>
    <th colspan="2">Accuracy</th>
  </tr>
  <tr>
    <td>Before</td>
    <td>After</td>
  </tr>
  <tr>
    <td>LeNet-300-100</td>
    <td>MNIST</td>
    <td>784->161/300*(ip)->64/100(ip)->10(ip)</td>
    <td>98.48%</td>
    <td>98.54%</td>
  </tr>
  <tr>
    <td>LeNet-5</td>
    <td>MNIST</td>
    <td>1->9/20(conv)->18/50(conv)->65/500(ip)->10(ip)</td>
    <td>99.34%</td>
    <td>99.34%</td>
  </tr>
  <tr>
    <td>ConvNet</td>
    <td>CIFAR-10</td>
    <td>3->28/32(conv)->22/32(conv)->40/64(conv)->10(ip)</td>
    <td>81.38%</td>
    <td>81.52%</td>
  </tr>
  <tr>
    <td>NIN</td>
    <td>CIFAR-10</td>
    <td>3->117/192(conv)->81/160(conv)->62/96(conv)<br>->183/192(conv)->123/192(conv)->148/192(conv)<br>->91/192(conv)->54/192(conv)->10(conv)</td>
    <td>89.67%</td>
    <td>89.68%</td>
  </tr>
  <tr>
    <td>AlexNet</td>
    <td>ImageNet</td>
    <td>3->83/96(conv)->225/256(conv)->233/384(conv)<br>->238/384(conv)->253/256(conv)->3001/4096(ip)<br>->3029/4096(ip)->1000(ip)</td>
    <td>80.3%</td>
    <td>80.5%</td>
  </tr>
</table>

\* <# of nodes after pruning>/<# of original nodes>

## MNIST
I use the dataset reader provided by [torchvision](https://github.com/pytorch/vision).  
#### LeNet-300-100
To run the pruning, please run:
```bash
$ cd <ROOT>/MNIST/
$ bash prune.node.lenet_300_100.sh
```
This script includes following commands:
```bash
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
```

It first trains the original model and, then, applies node pruning (stage 0-2). After node pruning, the model will be retrained to retain the original accuracy (stage 3).
#### LeNet-5
To run the pruning:
```bash
$ cd <ROOT>/MNIST/
$ bash prune.node.lenet_5.sh
```
It first trains the original model and then apply node pruning. The pre-pruned model can be download [here](https://drive.google.com/open?id=0B-7I62GOSnZ8N09rQU9scEQ2WXc). Download it and put it in the directory of ```<ROOT>/MNIST/saved_models/```. To evaluate the pruned model:
```bash
$ python main.py --prune node --arch LeNet_5 --pretrained saved_models/LeNet_5.prune.node.5.pth.tar --evaluate
```

## CIFAR-10
The training dataset can be downloaded [here](https://drive.google.com/open?id=0B-7I62GOSnZ8Z0ZCVXFtVnFEaTg). Download and uncompress it to ```<ROOT>/CIFAR_10/data/```.
#### ConvNet
Tor run the pruning:
```bash
$ cd <ROOT>/CIFAR_10/
$ bash prune.node.convnet.sh
```
Pre-pruned model can be downloaded [here](https://drive.google.com/open?id=0B-7I62GOSnZ8YlBvR2FBbTRCdGM). Download it and put it in the directory of ```<ROOT>/CIFAR_10/saved_models/```. To evaluate the pruned model:
```bash
$ python main.py --prune node --pretrained saved_models/ConvNet.prune.node.4.pth.tar --evaluate
```

#### Network-in-Network (NIN)
Tor run the pruning:
```bash
$ cd <ROOT>/CIFAR_10/
$ bash prune.node.nin.sh
```
Pre-pruned model can be downloaded [here](https://drive.google.com/open?id=0B-7I62GOSnZ8Unl3eFotRlZJX0E). Download it and put it in the directory of ```<ROOT>/CIFAR_10/saved_models/```. To evaluate the pruned model:
```bash
$ python main.py --prune node --arch NIN --pretrained saved_models/NIN.prune.node.7.pth.tar --evaluate
```


## ImageNet
Tor run the pruning:
```bash
$ cd <ROOT>/ImageNet/
$ bash prune.node.alexnet.sh
```

Pre-pruned model can be downloaded [here](https://drive.google.com/open?id=0B-7I62GOSnZ8STFVUm5JSUY2Vjg). Download it and put it in the directory of ```<ROOT>/ImageNet/saved_models/```. To evaluate the pruned model:
```bash
$ python main.py --prune node --pretrained saved_models/AlexNet.prune.node.8.pth.tar --evaluate
```


# SIMD-Aware Weight Pruning
SIMD-aware weight pruning is provided in ```./SIMD_Aware_MNIST```. LeNet-300-100 and LeNet-5 on MNIST is tested. The example of LeNet-300-100 can be executed by
```bash
$ cd ./SIMD_Aware_MNIST/
$ bash prune.simd.lenet_300_100.sh
```
It will first train the network and then perform the SIMD-aware weight pruning with group width set to 8. It can remove 92.0% of the weights. The script of ```prune.simd.lenet_300_100.sh``` contains following instructions:
```bash
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
```
For LeNet-5, the experiment can be performed by run
```bash
$ bash prune.simd.lenet_5.sh
```
It will remove 96.8% of the weights in LeNet-5.

SIMD-aware weight pruning for other benchmark networks are under construction.

# Citation
Please cite Scalpel in your publications if it helps your research:
```
@inproceedings{yu2017scalpel,
  title={Scalpel: Customizing DNN Pruning to the Underlying Hardware Parallelism},
  author={Yu, Jiecao and Lukefahr, Andrew and Palframan, David and Dasika, Ganesh and Das, Reetuparna and Mahlke, Scott},
  booktitle={Proceedings of the 44th Annual International Symposium on Computer Architecture},
  pages={548--560},
  year={2017},
  organization={ACM}
}
```
