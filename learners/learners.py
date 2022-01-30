import torch, sys
from    torch import nn
from    torch import optim
from    torch.nn import functional as F
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.linear import Linear


class FullyConnected(nn.Module):

    def __init__(self, n_way, k_shot):
        super(FullyConnected, self).__init__()
        self.n_way = n_way
        self.k_shot = k_shot
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, 3, 2, 0),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, 2, 0),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, 2, 0),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 2, 1, 0),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Flatten(),
            nn.Linear(64, 64 * 5),
            nn.Tanh()         
        )

    def forward(self, x):
        landa_param = self.model(x)
        return landa_param


class MetaLearner(nn.Module):
    def __init__(self, n_way):
        super(MetaLearner, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, 3, 2, 0),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, 2, 0),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, 2, 0),
            nn.ReLU(),
            nn.BatchNorm2d(64), 
            nn.Conv2d(64, 64, 2, 1, 0),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Flatten(),
            nn.Linear(64, n_way)
        )

    def forward(self, x):
        predict = self.model(x)
        return predict



class ConvNetwork(nn.Module):
    def __init__(self, n_way, k_shot):
        super(ConvNetwork, self).__init__()
        self.n_way = n_way
        self.k_shot = k_shot
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, 3, 2, 0),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 0),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 0),
            nn.ReLU(),
            # nn.Conv2d(64, 32, 3, 1, 0),
            nn.Flatten(),
            # nn.Linear(64 * 5)
        )

    def forward(self, x):
        x = self.model(x)
        print(x.size())
        sys.exit()


class InnerSumLearner(nn.Module):
    def __init__(self, n_way, k_shot):
        super(InnerSumLearner, self).__init__()
        self.n_way = n_way
        self.k_shot = k_shot
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, 3, 2, 0),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, 2, 0),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, 2, 0),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 2, 1, 0),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Flatten(),
            nn.Linear(64, 5),
            # nn.Tanh()         
        )

    # meta_param, meta_learner
    def forward(self, x, meta_param, meta_learner):  
    
        # self.meta_param = meta_param
        # self.meta_learner = meta_learner
        # self.meta_learner.eval()
        # self.landa_param = self.model(x)


        # self.landa_param = self.landa_param.mean(axis=0).view(5, 64)
        
        # # theta_param = self.meta_param[16] + self.landa_param

        # self.meta_param[16] = nn.Parameter(self.landa_param, requires_grad = True)
        # self.__set_meta_learner_param()

        # pred = self.meta_learner(x)
        pred = self.meta_learner(x)

        return pred
        
        
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
       