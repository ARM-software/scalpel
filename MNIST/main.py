from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import sys
cwd = os.getcwd()
sys.path.append(cwd+'/../')
import models
from torchvision import datasets, transforms
from torch.autograd import Variable
from newLayers import *
from util import *

def save_state(model, acc):
    print('==> Saving model ...')
    state = {
            'acc': acc,
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
        torch.save(state, 'saved_models/'+args.arch+'.best_origin.pth.tar')

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if args.prune == 'node':
            beta_penalty_op.penalize()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))
    return

def test(evaluate=False):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0

    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += criterion(output, target).data[0]
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    
    acc = 100. * correct / len(test_loader.dataset)
    if ((args.prune == 'node') and (not args.retrain)) or (acc > best_acc):
        best_acc = acc
        if not evaluate:
            save_state(model, best_acc)

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss * args.batch_size, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    print('Best Accuracy: {:.2f}%\n'.format(best_acc))
    return

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 15 epochs"""
    lr = args.lr * (0.1 ** (epoch // args.lr_epochs))
    print('Learning rate:', lr)
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

if __name__=='__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
            help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
            help='input batch size for testing (default: 128)')
    parser.add_argument('--epochs', type=int, default=60, metavar='N',
            help='number of epochs to train (default: 60)')
    parser.add_argument('--lr-epochs', type=int, default=15, metavar='N',
            help='number of epochs to decay the lr (default: 15)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
            help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
            help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
            metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
            help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
            help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
            help='how many batches to wait before logging training status')
    parser.add_argument('--arch', action='store', default='LeNet_300_100',
            help='the MNIST network structure: LeNet_300_100 | LeNet_5')
    parser.add_argument('--pretrained', action='store', default=None,
            help='pretrained model')
    parser.add_argument('--evaluate', action='store_true', default=False,
            help='whether to run evaluation')
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
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # check options
    if not (args.prune_target in [None, 'conv', 'ip']):
        print('ERROR: Please choose the correct prune_target')
        exit()

    print_args(args)
    
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    
    # load data
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data', train=True, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                    ])),
                batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
                ])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
    
    # generate the model
    if args.arch == 'LeNet_300_100':
        model = models.LeNet_300_100(args.prune)
    elif args.arch == 'LeNet_5':
        model = models.LeNet_5(args.prune)
    else:
        print('ERROR: specified arch is not suppported')
        exit()

    if not args.pretrained:
        best_acc = 0.0
    else:
        pretrained_model = torch.load(args.pretrained)
        best_acc = pretrained_model['acc']
        load_state(model, pretrained_model['state_dict'])

    if args.cuda:
        model.cuda()
    
    print(model)
    param_dict = dict(model.named_parameters())
    params = []
    
    base_lr = 0.1
    
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

    criterion = nn.CrossEntropyLoss()

    if args.evaluate:
        print_layer_info(model)
        test(evaluate=True)
        exit()

    if args.prune == 'node':
        print('==> Start node pruning ...')
        beta_penalty_op = beta_penalty(model, args.penalty, args.lr, args.prune_target)
        if not args.pretrained:
            print('==> ERROR: Please assign the pretrained model')
            exit()
        for epoch in range(1, args.epochs + 1):
            lr = adjust_learning_rate(optimizer, epoch)
            if args.retrain:
                beta_penalty_op.update_learning_rate(0.0)
            else:
                beta_penalty_op.update_learning_rate(lr)
            train(epoch)
            print_layer_info(model)
            test()
    else:
        for epoch in range(1, args.epochs + 1):
            adjust_learning_rate(optimizer, epoch)
            train(epoch)
            test()
