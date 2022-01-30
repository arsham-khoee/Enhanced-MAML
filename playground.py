import torch, os
# from learners import FullyConnected
from omniglotNShot import OmniglotNShot
import  argparse
from fixed_maml import FixedMaml
from torch import optim
from torch.nn import functional as F
import numpy as np
import sys

def main(args):
    # meta learner 
    config = [
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
        ('linear', [args.n_way, 64])
    ]
    # model
    model = FixedMaml(args, config).to('cuda')
    opt = optim.SGD(model.parameters(), lr = args.update_lr)
    # dataset
    db_train = OmniglotNShot('omniglot',
                       batchsz=args.task_num,
                       n_way=args.n_way,
                       k_shot=args.k_spt,
                       k_query=args.k_qry,
                       imgsz=args.imgsz)

    device = torch.device('cuda')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu) 
    final_acc, final_loss = [], []
    for step in range(40):
        x_spt, y_spt, x_qry, y_qry = db_train.next()
        x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).to(device), torch.from_numpy(y_spt).to(device), \
                                        torch.from_numpy(x_qry).to(device), torch.from_numpy(y_qry).to(device)
        accs = []
        losses = []
        for x_spt_one, y_spt_one, x_qry_one, y_qry_one in zip(x_spt, y_spt, x_qry, y_qry):
            print('********* here **********')
            print(x_spt_one.requires_grad)
            test_acc, test_loss = model(x_spt_one, y_spt_one, x_qry_one, y_qry_one)
            accs.append( test_acc[5] )
            # print('test acc')
            # print(test_acc)
            losses.append(test_loss.item())
        # print('loss')
        # print(losses)
        # print('accs')
        # print(accs)

        accs = np.array(accs).mean(axis=0).astype(np.float16)
        losses = np.array(losses).mean(axis=0).astype(np.float16)
        final_acc.append(accs)
        final_loss.append(losses)
        print('********* accs *********')
        print(accs)
    
    final_acc = np.array(final_acc).mean().astype(np.float16)
    final_loss = np.array(final_loss).mean().astype(np.float16)
    print(f' Final acc: {final_acc}')
    print(f' Final loss: {final_loss}')







if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--gpu', type=int, help='gpu cuda number', default=0)
    argparser.add_argument('--epoch', type=int, help='epoch number', default=40000)
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=5)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=15)
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=28)
    argparser.add_argument('--imgc', type=int, help='imgc', default=1)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=32)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.1)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=5)

    args = argparser.parse_args()

    main(args)