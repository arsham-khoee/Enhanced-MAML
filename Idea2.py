import torch, sys
from torch import nn
from torch import optim
from torch.nn import functional as F
import numpy as np
from torch import optim
# from copy import ddepcopy
from learners import FullyConnected, ConvNetwork,  InnerSumLearner, MetaLearner
# from learner import Learner
import copy


class Idea2(nn.Module):
    def __init__(self, args, config):
        super(Idea2, self).__init__()
        self.args = args
        # self.config = config
        self.update_lr = args.update_lr
        self.n_way = args.n_way
        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.task_num = args.task_num
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test

        # self.learner = FullyConnected(n_way=self.n_way, k_shot=self.k_spt).to('cuda')
        # self.meta_learner = Learner(config, args.imgc, args.imgsz)
        self.meta_learner = MetaLearner(n_way = args.n_way)
 
        # self.opt_learner = optim.Adam(self.learner.parameters(), lr = self.update_lr)
        self.meta_param, self.meta_bn_param = self.__load_meta_param()

        self.__set_meta_learner_param()
        
        self.learner_config = [
            ('conv2d', [64, 1, 3, 3, 2, 0]),
            ('relu', [True]),
            ('bn', [64]),
            ('conv2d', [64, 64, 3, 3, 2, 0]),
            ('relu', [True]),
            ('bn', [64]),
            ('conv2d', [64, 64, 3, 3, 2, 0]),
            ('relu', [True]),
            ('bn', [64]),
            ('conv2d', [64, 64, 2, 2, 1, 0]),
            ('relu', [True]),
            ('bn', [64]),
            ('flatten', []),
            ('linear', [args.n_way * 64, 64])
        ]
        

    def forward(self, x_spt, y_spt, x_qry, y_qry):

        deciaml_round = 5
        self.meta_param, _ = self.__load_meta_param()

        self.learner = InnerSumLearner(n_way=self.n_way, k_shot=self.k_spt).to('cuda')
        self.opt_learner = optim.Adam(self.learner.parameters(), lr = self.update_lr)
        
        querysz = x_qry.size(0)

        corrects = [0 for _ in range(self.update_step_test + 1)]
        losses_q = [0 for _ in range(self.update_step_test + 1)]

        net = self.meta_learner



        for k in range(self.update_step_test):
            print(k)
            self.meta_param, _ = self.__load_meta_param()
            # self.__set_meta_learner_param()
            self.learner.zero_grad()

            logits = self.learner(x_spt, self.meta_param, self.meta_learner)
            # landa_param = landa_param.mean(axis=0).view(5, 64)
            # theta_param = self.meta_param[16] + landa_param

            # self.meta_param[16] = nn.Parameter(theta_param, requires_grad = True)

            # self.__set_meta_learner_param()
            # self.meta_learner.eval()
            
            # logits = self.meta_learner(x_spt)
          
            # logits = self.learner(x_spt, self.meta_param, self.meta_learner)
            loss = F.cross_entropy(logits, y_spt)

            loss.backward()
           
            self.opt_learner.step()

            for idx, param in enumerate(self.learner.parameters()):
                # print(idx)
                if idx == 16:
                    print(param.size())
                    print(param)
            # sys.exit()
        

            logits_q = self.meta_learner(x_qry)
            loss_q = F.cross_entropy(logits_q, y_qry)
            losses_q[k + 1] += round(loss_q.item(), deciaml_round)

            with torch.no_grad():
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry).sum().item() 
                corrects[k + 1] = corrects[k + 1] + correct
       
        accs = np.array(corrects) / querysz
        
        print('acc')
        print(accs)
        return corrects, losses_q


    def __load_meta_param(self):
        maml = torch.load('./logs/maml/omniglot_cnn__learner_lr:0.4_meta_lr:0.001_task_num:32_update_step:5_epoch:40000_shot:5_way:5/4000')
        parameters = nn.ParameterList().to('cuda')
        meta_param = maml['params']
        bn_param = nn.ParameterList().to('cuda')
        for idx, param in enumerate(meta_param):
            if (param.find('bn') == -1):
                # print(idx)
                # parameters.append(nn.Parameter(meta_param[param]))
                if idx == 16:
                    parameters.append(nn.Parameter(meta_param[param]))
                else:
                    parameters.append(nn.Parameter(meta_param[param], requires_grad = False))
            else:
                bn_param.append(nn.Parameter(meta_param[param], requires_grad = False))
        return parameters, bn_param

    
    def __set_meta_learner_param(self):
            
        # print('*********** param *********')
        parameters = self.meta_learner.state_dict()
    
        idx = 0
        for named in parameters:
            if 'weight' in named:
                parameters[named] = self.meta_param[idx]
                idx += 1                
            if 'bias' in named:
                parameters[named] = self.meta_param[idx]
                idx += 1

        self.meta_learner.load_state_dict(parameters)
       

        # print(self.meta_param)
        # sys.exit()
