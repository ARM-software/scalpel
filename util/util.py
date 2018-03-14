from __future__ import print_function
import os
import sys 
import torch
import torch.nn as nn
cwd = os.getcwd()
sys.path.append(cwd+'/../')
from newLayers import *

def print_layer_info(model):
    index = 0
    print()
    for m in model.modules():
        if hasattr(m, 'alpha'):
            print('MaskLayer', index, ':',
                    m.alpha.data.nelement()-int(m.alpha.data.eq(1.0).sum()), 'of',
                    m.alpha.data.nelement(), 'is blocked')
            index += 1
    print()
    return

def print_args(args):
    print('\n==> Setting params:')
    for key in vars(args):
        print(key, ':', getattr(args, key))
    print('====================\n')
    return

class beta_penalty():
    def __init__(self, model, penalty, lr, prune_target):
        self.penalty = float(penalty)
        self.lr = float(lr)
        self.penalty_target = []
        for m in model.modules():
            if isinstance(m, MaskLayer):
                if prune_target:
                    if (prune_target == 'ip') and (m.conv == False):
                        self.penalty_target.append(m.beta)
                    if (prune_target == 'conv') and (m.conv == True):
                        self.penalty_target.append(m.beta)
                else:
                    self.penalty_target.append(m.beta)
        return

    def update_learning_rate(self, lr):
        self.lr = float(lr)
        return

    def penalize(self):
        for index in range(len(self.penalty_target)):
            self.penalty_target[index].data.sub_(self.penalty*self.lr)
        return

def load_state(model, state_dict):
    param_dict = dict(model.named_parameters())
    state_dict_keys = state_dict.keys()
    cur_state_dict = model.state_dict()
    for key in cur_state_dict:
        if key in state_dict_keys:
            cur_state_dict[key].copy_(state_dict[key])
        elif key.replace('module.','') in state_dict_keys:
            cur_state_dict[key].copy_(state_dict[key.replace('module.','')])
        elif 'module.'+key in state_dict_keys:
            cur_state_dict[key].copy_(state_dict['module.'+key])
    return

class dropout_update():
    def __init__(self, dropout_list, mask_list):
        self.dropout_list = dropout_list
        self.mask_list = mask_list
        self.dropout_param_list = []
        for i in range(len(self.dropout_list)):
            self.dropout_param_list.append(self.dropout_list[i].p)
        return

    def update(self):
        for i in range(len(self.dropout_list)):
            mask = self.mask_list[i].alpha.data
            dropout_tmp_value = float(mask.eq(1.0).sum()) / float(mask.nelement())
            dropout_tmp_value = dropout_tmp_value * self.dropout_param_list[i]
            self.dropout_list[i].p = dropout_tmp_value
        return

class SIMD_Prune_Op():
    def __init__(self, model, threshold, width, evaluate=False):
        if not evaluate:
            model.weights_pruned = []
            for m in model.modules():
                if isinstance(m, nn.Linear):
                    tmp_pruned = m.weight.data.clone()
                    append_size = width - tmp_pruned.shape[1]%width
                    tmp_pruned = torch.cat((tmp_pruned, tmp_pruned[:, 0:append_size]), 1)
                    tmp_pruned = tmp_pruned.view(tmp_pruned.shape[0], -1, width)
                    shape = tmp_pruned.shape
                    tmp_pruned = tmp_pruned.pow(2.0).mean(2, keepdim=True).pow(0.5)\
                            .expand(tmp_pruned.shape).lt(threshold)
                    tmp_pruned[:, -1] = False
                    tmp_pruned = tmp_pruned.view(tmp_pruned.shape[0], -1)
                    tmp_pruned = tmp_pruned[:, 0:m.weight.data.shape[1]]
                    model.weights_pruned.append(tmp_pruned)
                elif isinstance(m, nn.Conv2d):
                    tmp_pruned = m.weight.data.clone()
                    original_size = tmp_pruned.size()
                    tmp_pruned = tmp_pruned.view(original_size[0], -1)
                    append_size = width - tmp_pruned.shape[1]%width
                    tmp_pruned = torch.cat((tmp_pruned, tmp_pruned[:, 0:append_size]), 1)
                    tmp_pruned = tmp_pruned.view(tmp_pruned.shape[0], -1, width)
                    shape = tmp_pruned.shape
                    tmp_pruned = tmp_pruned.pow(2.0).mean(2, keepdim=True).pow(0.5)\
                            .expand(tmp_pruned.shape).lt(threshold)
                    tmp_pruned[:, -1] = False
                    tmp_pruned = tmp_pruned.view(original_size[0], -1)
                    tmp_pruned = tmp_pruned[:, 0:m.weight.data[0].nelement()]
                    tmp_pruned = tmp_pruned.view(original_size)
                    model.weights_pruned.append(tmp_pruned)
        self.weights_pruned = model.weights_pruned
        self.model = model
        self.print_info()
        self.prune_weight()
        return
    
    def prune_weight(self):
        index = 0
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                m.weight.data[self.weights_pruned[index]] = 0
                index += 1
            elif isinstance(m, nn.Conv2d):
                m.weight.data[self.weights_pruned[index]] = 0
                index += 1
        return
    
    def print_info(self):
        print('------------------------------------------------------------------')
        print('- SIMD-aware weight pruning info:')
        pruned_acc = 0
        total_acc = 0
        for i in range(len(self.weights_pruned)):
            pruned = int(self.weights_pruned[i].sum())
            total = int(self.weights_pruned[i].nelement())
            pruned_acc += pruned
            total_acc += total
            print('- Layer '+str(i)+': '+'{0:10d}'.format(pruned)+' / '+\
                    '{0:10d}'.format(total)+ ' ('\
                    '{0:4.1f}%'.format(float(pruned)/total * 100.0)+\
                    ') weights are pruned')
        print('- Total  : '+'{0:10d}'.format(pruned_acc)+' / '+\
                '{0:10d}'.format(total_acc)+ ' ('\
                '{0:4.1f}%'.format(float(pruned_acc)/total_acc * 100.0)+\
                ') weights are pruned')
        print('------------------------------------------------------------------\n')
        return
