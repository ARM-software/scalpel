import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.optim as optim
from models import *
import sys
import gc
cwd = os.getcwd()
sys.path.append(cwd+'/../')
import datasets as datasets
import datasets.transforms as transforms
from util import *
from newLayers import *
import time

os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"

# set the seed
print("==> set seed:",long(time.time()))
torch.manual_seed(long(time.time()))
torch.cuda.manual_seed(long(time.time()))

def save_state(model, prec1, prec5):
    print('==> Saving mode ...')
    state = {
            'prec1':prec1,
            'prec5':prec5,
            'state_dict': model.state_dict(),
            }
    for key in state['state_dict'].keys():
        if 'module' in key:
            state['state_dict'][key.replace('module.', '')] = \
                    state['state_dict'].pop(key)
    if args.prune == 'node':
        torch.save(state, 'saved_models/'+args.arch+'.prune.'+\
                args.prune+'.'+str(args.stage)+'.pth.tar')
    else:
        # torch.save(state, 'saved_models/'+args.arch+'.best_origin.pth.tar')
        print('==> Currently the embedded torch save func is used')
        exit()

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--arch', '-a', metavar='ARCH', default='AlexNet',
        help='model architecture (default: AlexNet)')
parser.add_argument('--data', metavar='DATA_PATH', default='./data/',
        help='path to imagenet data (default: ./data/)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
        help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
        help='number of total epochs to run')
parser.add_argument('--lr-epochs', type=int, default=25, metavar='N',
        help='number of epochs to decay the lr (default: 25)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
        help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
        metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
        metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.90, type=float, metavar='M',
        help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
        metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
        metavar='N', help='print frequency (default: 10)')
parser.add_argument('--print-freq-mask', '--pm', default=10000, type=int,
        metavar='N', help='print frequency of the mask info (default: 10000)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
        help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
        help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store',
        default=None, help='use pre-trained model')
parser.add_argument('--retrain', action='store_true', default=False,
        help='retrain the pruned network')
parser.add_argument('--prune', action='store', default=None,
        help='pruning mechanism: node')
parser.add_argument('--prune-target', action='store', default=None,
        help='pruning target: default=None | conv | ip')
parser.add_argument('--stage', action='store', type=int, default=0,
        help='pruning stage')
parser.add_argument('--penalty', action='store', default=0.0,
        help='beta penalty')
parser.add_argument('--beta-initial', action='store', type=float, default=0.8002,
        help='initial value of beta')
parser.add_argument('--beta-limit', action='store', type=float, default=0.802,
        help='upper-bound value of beta')

best_prec1 = 0
args = parser.parse_args()

def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if args.prune == 'node':
            beta_penalty_op.penalize()
            if args.arch == 'AlexNet':
                dropout_update_op.update()
                # avoid overfit
                for m in model.modules():
                    if isinstance(m, nn.Dropout):
                        tmp_p = m.p / 0.5
                        tmp_p = tmp_p**0.5
                        m.p = tmp_p * 0.5

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
        if (i % args.print_freq_mask == 0) and (args.prune=='node'):
            print_layer_info(model)
            for m in model.modules():
                if isinstance(m, MaskLayer):
                    print(m.beta.data.min())
        gc.collect()


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, top5.avg


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    lr = args.lr * (0.1 ** (epoch // args.lr_epochs))
    print 'Learning rate:', lr
    for param_group in optimizer.param_groups:
        if args.retrain and ('mask' in param_group['key']): # retraining
            param_group['lr'] = 0.0
        elif args.prune_target and ('mask' in param_group['key']):
            if args.prune_target in param_group['key']:
                param_group['lr'] = lr
            else:
                param_group['lr'] = 0.0
        else:
            param_group['lr'] = lr
    return lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    # check options
    if not (args.prune_target in [None, 'conv', 'ip']):
        print('ERROR: Please choose the correct prune_target')
        exit()

    print_args(args)

    # create model
    if args.arch=='AlexNet':
        model = AlexNet(prune=args.prune,
                beta_initial=args.beta_initial, beta_limit=args.beta_limit)
        input_size = 227
    else:
        raise Exception('Model not supported yet')

    if args.arch == 'AlexNet':
        model.cuda()
        model = torch.nn.DataParallel(model)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    
    param_dict = dict(model.named_parameters())
    params = []

    for key, value in param_dict.items():
        if 'mask' in key:
            params += [{'params':[value], 'lr': args.lr,
                'momentum':args.momentum,
                'weight_decay': 0.0,
                'key':key}]
        else:
            params += [{'params':[value], 'lr': args.lr,
                'momentum':args.momentum,
                'weight_decay': args.weight_decay,
                'key':key}]
    
    optimizer = optim.SGD(params, lr=args.lr, momentum=args.momentum,
            weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            exit()
    elif args.pretrained:
        if os.path.isfile(args.pretrained):
            print("==> load pretrained model '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained)
            load_state(model, checkpoint['state_dict'])
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrained))
            exit()
    else:
        # initialization
        for m in model.modules():
            if isinstance(m, nn.Conv2d): 
                m.weight.data.normal_(0.0, 0.05)
                m.bias.data.zero_().add(1.0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.01)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.zero_().add_(1.0)
                m.bias.data.zero_()

    cudnn.benchmark = True

    # Data loading code
    if not os.path.exists(args.data+'/imagenet_mean.binaryproto'):
        print("==> Data directory"+args.data+"does not exits")
        print("==> Please specify the correct data path by")
        print("==>     --data <DATA_PATH>")
        exit()

    normalize = transforms.Normalize(
            meanfile=args.data+'/imagenet_mean.binaryproto')

    train_dataset = datasets.ImageFolder(
        args.data,
        transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            transforms.RandomSizedCrop(input_size),
        ]),
        Train=True)

    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(args.data, transforms.Compose([
            transforms.ToTensor(),
            normalize,
            transforms.CenterCrop(input_size),
        ]),
        Train=False),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    print model

    if args.evaluate:
        print_layer_info(model)
        validate(val_loader, model, criterion)
        exit()

    if args.prune == 'node':
        print('==> Start node pruning ...')
        beta_penalty_op = beta_penalty(model, args.penalty, args.lr, args.prune_target)
        if args.arch == 'AlexNet':
            dropout_update_op = dropout_update(
                    [model.module.dropout1, model.module.dropout2],
                    [model.module.mask_ip6, model.module.mask_ip7])
        if not args.pretrained:
            print('==> ERROR: Please assign the pretrained model')
            exit()
        for epoch in range(args.start_epoch, args.epochs):
            lr = adjust_learning_rate(optimizer, epoch)
            if args.retrain:
                beta_penalty_op.update_learning_rate(0.0)
            else:
                beta_penalty_op.update_learning_rate(lr)

            # train for one epoch
            train(train_loader, model, criterion, optimizer, epoch)
            print_layer_info(model)

            # evaluate on validation set
            prec1, prec5 = validate(val_loader, model, criterion)
            save_state(model, prec1, prec5)
    else:
        for epoch in range(args.start_epoch, args.epochs):
            adjust_learning_rate(optimizer, epoch)

            # train for one epoch
            train(train_loader, model, criterion, optimizer, epoch)

            # evaluate on validation set
            prec1, prec5 = validate(val_loader, model, criterion)

            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            if is_best:
                filename = 'saved_models/'+args.arch+'.best_original.pth.tar'
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_prec1': best_prec1,
                    # 'optimizer' : optimizer.state_dict(),
                }, filename=filename)
