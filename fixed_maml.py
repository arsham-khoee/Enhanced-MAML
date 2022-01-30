import torch, sys
from torch import nn
from torch import optim
from torch.nn import functional as F
import numpy as np
from torch import optim
# from copy import ddepcopy
from learners import FullyConnected
from learner import Learner
from    copy import deepcopy


class FixedMaml(nn.Module):
    def __init__(self, args, config):
        super(FixedMaml, self).__init__()
        self.args = args
        self.update_lr = args.update_lr
        self.n_way = args.n_way
        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.task_num = args.task_num
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test

        self.learner = FullyConnected(n_way=self.n_way, k_shot=self.k_spt).to('cuda')
        self.meta_learner = Learner(config, args.imgc, args.imgsz)
        self.optim = optim.SGD(self.learner.parameters(), lr = self.update_lr)
        self.__load_meta_param()
        # self.meta_learner.load_state_dict(self.meta_param)
        # print(self.meta_param)
    
    def forward(self, x_spt, y_spt, x_qry, y_qry):
        
        # task_num, setsz, c_, h, w = x_spt.size()
        querysz = x_qry.size(0)

        # losses_q = [0 for _ in range(self.update_step + 1)] 
        corrects = [0 for _ in range(self.update_step_test + 1)]

        net = self.meta_learner
        # net.vars_bn = self.meta_bn_param

        logits = net(x_spt, vars=self.meta_param)
        loss = F.cross_entropy(logits, y_spt)

        grad = torch.autograd.grad(loss, self.meta_param, allow_unused=True)

        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.meta_param)))

        # this is the loss and accuracy before first update
        with torch.no_grad():
            
            logits_q = net(x_qry, self.meta_param, bn_training=True)
            
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[0] = corrects[0] + correct

        # this is the loss and accuracy after the first update
        with torch.no_grad():
            
            logits_q = net(x_qry, fast_weights, bn_training=True)
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[1] = corrects[1] + correct

        for k in range(1, self.update_step_test):
            # 1. run the i-th task and compute loss for k=1~K-1
            logits = net(x_spt, fast_weights, bn_training=True)
            loss = F.cross_entropy(logits, y_spt)
            # 2. compute grad on theta_pi
            grad = torch.autograd.grad(loss, fast_weights, allow_unused=True)
            # 3. theta_pi = theta_pi - train_lr * grad
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

            logits_q = net(x_qry, fast_weights, bn_training=True)
            # loss_q will be overwritten and just keep the loss_q on last update step.
            loss_q = F.cross_entropy(logits_q, y_qry)

            with torch.no_grad():
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry).sum().item()  # convert to numpy
                corrects[k + 1] = corrects[k + 1] + correct
       
        accs = np.array(corrects) / (querysz)
        print('accs')
        print(accs)
        return accs, loss


    def __load_meta_param(self):
        maml = torch.load('./logs/maml/omniglot_cnn__learner_lr:0.4_meta_lr:0.001_task_num:32_update_step:5_epoch:40000_shot:5_way:5/4000')
        parameters = nn.ParameterList().to('cuda')
        self.meta_param = maml['params']
        bn_param = nn.ParameterList().to('cuda')
        for param in self.meta_param:
            if (param.find('bn') == -1):
                print('param')
                print(self.meta_param[param])
                parameters.append(nn.Parameter(self.meta_param[param]))
            else:
                bn_param.append(nn.Parameter(self.meta_param[param], requires_grad=False))
        self.meta_param = parameters
        self.meta_bn_param = bn_param
        print(self.meta_param)