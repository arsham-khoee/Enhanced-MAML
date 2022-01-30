import torch, os, sys
from torch.utils.tensorboard.writer import SummaryWriter
# from learners import FullyConnected
from omniglotNShot import OmniglotNShot
import  argparse
from Idea import Idea
from torch import optim
from torch.nn import functional as F
import numpy as np
from Idea2 import Idea2
from Idea import Idea
import os.path as osp


def set_run_path(args):
    log_base_dir = './logs/'
    log_folder = 'maml'
    if not osp.exists(log_folder):
        os.mkdir(log_folder)
    meta_base_dir = osp.join(log_base_dir, log_folder)
    if not osp.exists(meta_base_dir):
        os.mkdir(meta_base_dir)
    save_path1 = '_'.join(['omniglot', 'cnn'])
    obj = {
        'learner_lr': args.update_lr, 
        'meta_lr': args.meta_lr,
        'task_num': args.task_num,
        'update_step': args.update_step, 
        'epoch': args.epoch,
        'shot': args.n_way,
        'way': args.k_spt,
    }
    save_path2 = ''
    for item in obj:
        save_path2 += f'_{str(item)}:{obj[item]}'
            
    save_path = f'{meta_base_dir}/{save_path1}_{save_path2}'
        # save_path = f'{save_path2}'
    if os.path.exists(save_path):
        pass
    else:
        os.mkdir(save_path)
        # return save_path
    print(save_path)
    return save_path



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

    save_path = set_run_path(args)
    writer = SummaryWriter(comment = save_path)
    
    device = torch.device('cuda')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu) 
    print('Using gpu:', args.gpu)

    # model
    model = Idea2(args, config).to(device)
    opt = optim.SGD(model.parameters(), lr = args.update_lr)

    # dataset
    db_train = OmniglotNShot('omniglot',
                       batchsz=args.task_num,
                       n_way=args.n_way,
                       k_shot=args.k_spt,
                       k_query=args.k_qry,
                       imgsz=args.imgsz)


    final_acc, final_loss = [], []

    x_spt, y_spt, x_qry, y_qry = db_train.next()
    x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).to(device), torch.from_numpy(y_spt).to(device), \
                                        torch.from_numpy(x_qry).to(device), torch.from_numpy(y_qry).to(device)

    for step in range(1):
        accs = []
        losses = []
        # print(x_spt.size())
        task_num = 1
        for x_spt_one, y_spt_one, x_qry_one, y_qry_one in zip(x_spt, y_spt, x_qry, y_qry):
            # print(x_spt_one.size())
            print(task_num)
            task_num += 1
            test_acc, test_loss = model(x_spt_one, y_spt_one, x_qry_one, y_qry_one)
            # sys.exit()
            print('test acc')
            print(test_acc)
            # print('test loss')
            # print(test_loss)
            # sys.exit()

            # accs.append( test_acc[10] )
            # losses.append(test_loss.item())
        print('loss')
        print(losses)
        print('accs')
        print(accs)

        # accs = np.array(accs).mean(axis=0).astype( np.float16)
        # losses = np.array(losses).mean(axis=0).astype(np.float16)
        # final_acc.append(accs)
        # final_loss.append(losses)
        # print('********* accs *********')
        # print(accs)
    
    # final_acc = np.array(final_acc).mean().astype(np.float16)
    # final_loss = np.array(final_loss).mean().astype(np.float16)
    # print(f' Final acc: {final_acc}')
    # print(f' Final loss: {final_loss}')







if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--gpu', type=int, help='gpu cuda number', default=0)
    argparser.add_argument('--epoch', type=int, help='epoch number', default=40000)
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=5)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=15)
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=28)
    argparser.add_argument('--imgc', type=int, help='imgc', default=1)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=16)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.15)
    argparser.add_argument('--update_step', type=int, help='task-level inner update 28steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=150)

    args = argparser.parse_args()

    main(args)