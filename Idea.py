import torch, sys
from torch import nn
from torch import optim
from torch.nn import functional as F
import numpy as np
from torch import optim
# from copy import ddepcopy
from learners import FullyConnected, ConvNetwork
from learner import Learner
import copy


class Idea(nn.Module):
    def __init__(self, args, config):
        super(Idea, self).__init__()
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
        self.optimizer = torch.optim.SGD(self.learner.parameters(), lr=0.4)

    
    def forward(self, x_spt, y_spt, x_qry, y_qry):
        
        # task_num, setsz, c_, h, w = x_spt.size()
        # print(x_qry.size())
        querysz = x_qry.size(0)

        
        corrects = [0 for _ in range(50 + 1)]

        net = self.meta_learner
        # net.vars_bn = self.meta_bn_param

        landa_param = self.learner(x_spt)
        # print(landa_param.size())
        # sys.exit()
       
        landa_param = landa_param.mean(axis=0).view(5, 64)
        
        # print(landa_param.size())
        # print(landa_param)
        theta_param = self.meta_param[16] + landa_param
        # theta_param.requires_grad = False
        
        # print(theta_param.requires_grad)
        # grad = torch.autograd.grad(loss, self.meta_param, allow_unused=True)
       

        # fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.meta_param)))

        # theta_param.to('cuda')
        
        self.meta_param[16] = nn.Parameter(theta_param)
        # sys.exit()

        logits = net(x_spt, vars=self.meta_param)

        loss = F.cross_entropy(logits, y_spt)

        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()
       
        # grad = torch.autograd.grad(loss, self.meta_param, allow_unused=True)
       

        # fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.meta_param)))

        
        with torch.no_grad():
            
            logits_q = net(x_qry, self.meta_param , bn_training=True)
            
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[0] = corrects[0] + correct

        
        with torch.no_grad():
           
            logits_q = net(x_qry, self.meta_param, bn_training=True)
            
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            
            correct = torch.eq(pred_q, y_qry).sum().item()

            corrects[1] = corrects[1] + correct

        for k in range(1, 50):
            
            landa_param = self.learner(x_spt)
            landa_param = landa_param.mean(axis=0).view(5, 64)
            theta_param = self.meta_param[16] + landa_param
            
            self.meta_param[16] = nn.Parameter(theta_param.detach())


            logits = net(x_spt, self.meta_param, bn_training=True)
            loss = F.cross_entropy(logits, y_spt)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # grad = torch.autograd.grad(loss, fast_weights, allow_unused=True)
            
            # fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

            logits_q = net(x_qry, self.meta_param, bn_training=True)

            loss_q = F.cross_entropy(logits_q, y_qry)

            with torch.no_grad():
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry).sum().item() 
                corrects[k + 1] = corrects[k + 1] + correct
       
        accs = np.array(corrects) / querysz
        # print(querysz)
        # sys.exit()
        return accs, loss


    def __load_meta_param(self):
        maml = torch.load('./logs/maml/omniglot_cnn__learner_lr:0.4_meta_lr:0.001_task_num:32_update_step:5_epoch:40000_shot:5_way:5/9000')
        parameters = nn.ParameterList().to('cuda')
        self.meta_param = maml['params']
        bn_param = nn.ParameterList().to('cuda')
        for param in self.meta_param:
            if (param.find('bn') == -1):

                parameters.append(nn.Parameter(self.meta_param[param]))
            else:
                bn_param.append(nn.Parameter(self.meta_param[param], requires_grad=False))
        self.meta_param = parameters
        self.meta_bn_param = bn_param
        print(self.meta_param)
       