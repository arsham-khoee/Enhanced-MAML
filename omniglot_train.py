import  torch, os
import  numpy as np
from    omniglotNShot import OmniglotNShot
import  argparse
from torch.utils.tensorboard import SummaryWriter
from    meta import Meta
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


def save_model(model, save_path):
    torch.save(dict(params = model.state_dict()), save_path)


def main(args):

    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)

    print(args)

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

    maml = Meta(args, config).to(device)

    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(maml)
    print('Total trainable tensors:', num)                    

    db_train = OmniglotNShot('omniglot',
                       batchsz=args.task_num,
                       n_way=args.n_way,
                       k_shot=args.k_spt,
                       k_query=args.k_qry,
                       imgsz=args.imgsz)

    train_count, test_count = 0, 0
    for step in range(args.epoch):

        x_spt, y_spt, x_qry, y_qry = db_train.next()
        x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).to(device), torch.from_numpy(y_spt).to(device), \
                                     torch.from_numpy(x_qry).to(device), torch.from_numpy(y_qry).to(device)

        # set traning=True to update running_mean, running_variance, bn_weights, bn_bias
        accs, loss = maml(x_spt, y_spt, x_qry, y_qry)
        train_count += 1
        # print('here')
        # print(accs)
        # print('loss')
        # print(loss)
        maml.get_log(loss, accs)

        writer.add_scalar('data/loss', float(loss.item()), train_count)
        writer.add_scalar('data/acc', float(accs[4]), train_count)

        # save trlog
        torch.save(maml.trlog, osp.join(save_path, 'trlog'))

        if step % 50 == 0:
            save_model(maml, '{}/{}'.format(save_path, step))
            print('step:', step, '\ttraining acc:', accs)

        if step % 500 == 0:
            accs = []
            losses = []
            for _ in range(1000//args.task_num):
                # test
                x_spt, y_spt, x_qry, y_qry = db_train.next('test')
                x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).to(device), torch.from_numpy(y_spt).to(device), \
                                             torch.from_numpy(x_qry).to(device), torch.from_numpy(y_qry).to(device)

                # split to single task each time
                for x_spt_one, y_spt_one, x_qry_one, y_qry_one in zip(x_spt, y_spt, x_qry, y_qry):
                    
                    test_acc, test_loss = maml.finetunning(x_spt_one, y_spt_one, x_qry_one, y_qry_one)
                    accs.append( test_acc )
                    losses.append(test_loss.item())

                    maml.get_log(test_loss, test_acc, train = False)


            # [b, update_step+1]                                        
            accs = np.array(accs).mean(axis=0).astype(np.float16)
            # print(losses)
            losses = np.array(losses).mean(axis=0).astype(np.float16)
            print('Test acc:', losses)

            
            test_count += 1
            writer.add_scalar('data/test_loss', float(losses), test_count)
            writer.add_scalar('data/test_acc', float(accs[8]), test_count)
            print('loss', losses)
            print('Test acc:', accs)


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
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.4)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)

    args = argparser.parse_args()

    main(args)
