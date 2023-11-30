# -*- coding: utf-8 -*-
# @Author  : lan
# @Software: PyCharm
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1,2,3'

import sys

sys.path.append('..')

from models.model_all2 import *

from tqdm import tqdm
import time
import argparse

from torch.utils.tensorboard import SummaryWriter

import numpy as np
import torch_geometric
from torch.utils.data import Dataset  # , DataLoader
from torch_geometric.data import Data, DataListLoader  # DataLoader

from sklearn.metrics import recall_score, roc_auc_score, f1_score

from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument('--mode', type=str, help=" train or val")
parser.add_argument('--model_name', type=str, help="the model name", default='daslic2002_bS128_adam_lrdecay_Netchan5p1')
parser.add_argument('--gpu', type=str, help='gpu id', default='0,1,2')

# dataset
parser.add_argument('--num_classes', type=int, help="classification number", default=10)
parser.add_argument('--num_heads', type=int, help="classification number", default=8)

parser.add_argument('--data_type', type=str, default='cifar10')
parser.add_argument('--data_dir', type=str, help="the path of your train datasets",
                    default='./graph/cifar_h5/')
parser.add_argument('--num_features', type=int, help='the dim of graph features', default=14)

# training
parser.add_argument('--batch_size', type=int, help="batch size", default=128)
# parser.add_argument('--num_epochs',                    type=int,help='epochs',default=500)
parser.add_argument('--total_steps', type=int, help='the total iteration number, cifar is 15k, mnist is 20k', default=150000)
parser.add_argument('--num_workers', type=int, default=1)
# parser.add_argument('--learning_rate',             type=float, help='initial learning rate', default=0.1)
parser.add_argument('--lr_group', type=str, help='the lr group',
                    default='0.001,0.0005,0.0001,0.00005,0.00001')
parser.add_argument('--weight_decay', type=float, help='weight decay factor for optimization', default=1e-4)

# log and save
parser.add_argument('--checkpoint_path', type=str, help='path to a checkpoint to load', default='')
parser.add_argument('--log_directory', type=str, help='directory to save summaries', default='runs')
parser.add_argument('--log_freq', type=int, help='Logging frequency in global steps', default=50)
# online test

parser.add_argument('--test_freq', type=int, help='Online evaluation frequency in global steps', default=50)
parser.add_argument('--patience', type=int, help='patience times to adjust lr if test acc can not be better',
                    default=150)

parser.add_argument('--checkpoint_directory', type=str, help='directory to save model', default='checkpoints')

args = parser.parse_args()

print('args', args)

class MyScheduler:
    def __init__(self, lr_groups: list, ):

        self.lr_groups = lr_groups

    def update_lr(self, optimizer):
        # if len(self.lr_groups)!=1:
        #
        #
        #         lr=self.lr_groups.pop(0)
        # else:
        #     lr = self.lr_groups[0]
        for i, param_group in enumerate(optimizer.param_groups):
            if len(self.lr_groups[i]) != 1:
                param_group['lr'] = self.lr_groups[i].pop(0)
            else:
                param_group['lr'] = self.lr_groups[i][0]


# using for splitting the str gpu idxes and put them into a list

def str2list(str, type: str):
    '''

    Args:
        str: the input str
        type: the type you want to change

    Returns: list[type]

    '''
    assert type in ['int', 'float', 'str'], 'type choose from int ,float or str'
    data_list = []
    str_list = str.split(',')
    if type == 'int':
        for str in str_list:
            data_list.append(int(str))
    elif type == 'float':
        for str in str_list:
            data_list.append(float(str))
    elif type == 'str':
        data_list = str_list
    return data_list


def online_test(model, dataloader_test, criterion):
    correct = 0
    test_loss = 0.0
    test_total = 0

    y_true = []
    y_pred = []
    y_score = []
    for _, test_sample_batched in enumerate(dataloader_test):
        with torch.no_grad():
            labels_test = [sb.y for sb in test_sample_batched]
            labels_test = torch.stack(labels_test).cuda()

            # outputs, att = net(data)
            outputs = model(test_sample_batched)
            outputs_softmax = torch.nn.functional.softmax(outputs, dim=1)

            loss = criterion(outputs, labels_test.long())

            test_loss += loss.item()
            test_total += labels_test.size(0)
            pred = outputs.argmax(dim=1)

            y_pred.extend(pred.cpu().numpy())
            y_true.extend(labels_test.cpu().numpy())
            y_score.extend(outputs_softmax.cpu().numpy())

            correct += (pred == labels_test).sum().cuda()

    test_loss = test_loss / test_total
    test_acc = 100 * correct / test_total
    return test_loss, test_acc, y_pred, y_true, y_score



import os
import h5py
import torch.utils.data as data


class MyDataset(Dataset):
    def __init__(self, data_dir, split='train'):
        self.data_dir = data_dir
        self.split = split
        self.filenames = os.listdir(os.path.join(self.data_dir, self.split))

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        with h5py.File(os.path.join(self.data_dir, self.split, filename), 'r') as f:
            x = torch.as_tensor(np.array(f['x'])[:, :14], dtype=torch.float)  # node features
            edge_index = torch.as_tensor(np.array(f['edge_index']) - 1, dtype=torch.long)  # adjacency matrix
            edge_attr = torch.as_tensor(np.array(f['edge_attr']), dtype=torch.float)  # edge features
            y = torch.as_tensor(np.array(f['y']), dtype=torch.int)  # labels
            sample = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        return sample

    def __len__(self):
        return len(self.filenames)


# loading datasets

dataset = MyDataset(args.data_dir, split='train')
dataset_test = MyDataset(args.data_dir, split='test')

# DataLoader

dataloader = DataListLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
dataloader_test = DataListLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=0)

print('length of: train_dataset', len(dataset))
print('length of: test_dataset', len(dataset_test))


def train(args):
    print("total {} GPUs".format(torch.cuda.device_count()))
    random_state = 1
    torch.manual_seed(random_state)
    torch.cuda.manual_seed(random_state)
    torch.cuda.manual_seed_all(random_state)
    np.random.seed(random_state)
    torch.set_num_threads(2)
    torch.cuda.empty_cache()

    use_gpu = torch.cuda.is_available()
    print('args.gpu', args.gpu)
    gpu_group = str2list(args.gpu, type='int')
    print('gpu_group', gpu_group)
    print("*" * 15, "\nif cuda available:{},and will use gpu:{}".format(use_gpu, args.gpu))

    # init  model
    model = Net_chan5p4(in_feats=args.num_features, num_heads=args.num_heads, num_class=args.num_classes)
    global_step = 0
    total_steps = args.total_steps
    best_test_acc = 0
    best_test_steps = 0
    pati = 0
    criterion = torch.nn.CrossEntropyLoss()

    base_lr = str2list(args.lr_group, type='float')
    lr_group = [base_lr]
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr.pop(0), betas=(0.9, 0.999), eps=1e-8)

    scheduler = MyScheduler(lr_groups=lr_group)
    model_part_name = ['base']

    print(model)
    num_params = sum(p.numel() for p in model.parameters())
    print("== Total number of parameters: {}".format(num_params))

    num_params_update = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print("== Total number of learning parameters: {}".format(num_params_update))
    for name, child in model.named_children():
        params = sum(p.numel() for p in child.parameters())
        print('== {}  parameters:{}'.format(name, params))

    model = model.cuda()
    model = torch_geometric.nn.DataParallel(model)
    model.train()

    print("== Model Initialized")

    print("*" * 10, 'start build dataloader', "*" * 10)

    print("train data number:{},test data number:{}".format(len(dataset), len(dataset_test)))
    train_writer = SummaryWriter(args.log_directory + '/' + args.model_name + '/train')

    test_writter = SummaryWriter(args.log_directory + '/' + args.model_name + '/test')

    start_time = time.time()
    duration = 0

    if not os.path.isdir(args.checkpoint_directory + '/' + args.model_name):
        os.mkdir(args.checkpoint_directory + '/' + args.model_name)

    while global_step < total_steps:

        print('*' * 10, 'start training')
        if pati >= args.patience:
            pati = 0
            scheduler.update_lr(optimizer=optimizer)
            print('--------------------Reducing learning rate--------------------------')
        for step, sample_batched in enumerate(dataloader):
            model.train()

            optimizer.zero_grad()

            before_op_time = time.time()

            train_labels = [sb.y for sb in sample_batched]
           
            train_labels = torch.stack(train_labels).cuda()

            outputs = model(sample_batched)

            loss = criterion(outputs, train_labels.long())
            loss.backward()

            pred = outputs.argmax(dim=1)

            train_correct = (pred == train_labels).sum().cuda()
            train_acc = train_correct / train_labels.size(0)
            current_lr = []
            for i, name in enumerate(model_part_name):
                current_lr.append(optimizer.state_dict()['param_groups'][i]['lr'])
            optimizer.step()

            print(
                '[gobal step/total steps]: [{}/{}], base_lr: {:.6f}, loss: {:.8f}, train acc：{:.8f}'.format(global_step,
                                                                                                            total_steps,
                                                                                                            current_lr[
                                                                                                                0],
                                                                                                            loss,
                                                                                                            train_acc))
            if np.isnan(loss.cpu().item()):
                print('NaN in loss occurred. Aborting training.')
                return -1
            duration += time.time() - before_op_time

            if global_step and global_step % args.log_freq == 0:  # record the training log every 'args.log_freq' steps

                examples_per_sec = args.batch_size / duration * args.log_freq
                duration = 0
                time_sofar = (time.time() - start_time) / 3600
                training_time_left = (total_steps / global_step - 1.0) * time_sofar

                print_string = ' examples/s: {:4.2f} | loss: {:.5f} | time elapsed: {:.2f}h | time left: {:.2f}h'
                print(print_string.format(examples_per_sec, loss, time_sofar, training_time_left))

                train_writer.add_scalar('train_loss/step', loss, global_step)
                train_writer.add_scalar('train_acc/step', train_acc, global_step)
                for i, name in enumerate(model_part_name):
                    train_writer.add_scalar('lr of {}'.format(name), current_lr[i], global_step)
                train_writer.flush()

            if global_step % args.test_freq == 0: 
                model.eval()

                with torch.no_grad():
                    test_loss, test_acc, y_pred, y_true, outputs = online_test(model, dataloader_test,
                                                                               criterion)  # 记录20步测试的日志
                    test_writter.add_scalar('test_loss/step', test_loss, global_step)
                    test_writter.add_scalar('test_accuracy/step', test_acc, global_step)

                    # computing Recall、AUC 和 F1-score
                    recall = recall_score(y_true, y_pred, average='macro')  # Recall
                    auc = roc_auc_score(y_true, outputs, multi_class='ovr')  # AUC
                    f1_sc = f1_score(y_true, y_pred, average='macro')  # F1-score

                    print(f'Recall: {recall:.8f}, AUC: {auc:.8f}, f1_score:{f1_sc:.8f}')
                    test_writter.add_scalar('Recall/step', recall, global_step)
                    test_writter.add_scalar('AUC/step', auc, global_step)
                    test_writter.add_scalar('f1_score/step', f1_sc, global_step)

                    is_best = False

                    if test_acc > best_test_acc:
                        old_best = best_test_acc 
                        best_test_acc = test_acc
                        is_best = True

                    if is_best:
                        pati = 0
                        old_best_step = best_test_steps
                        old_best_name = '/model-{}-best_{}_{:.5f}'.format(old_best_step, 'test_acc',
                                                                          old_best)
                        model_path = args.checkpoint_directory + args.model_name + old_best_name
                        if os.path.exists(model_path):
                            command = 'rm {}'.format(model_path)
                            os.system(command)
                        best_test_steps = global_step
                        model_save_name = '/model-{}-best_{}_{:.5f}'.format(global_step, 'test_acc',
                                                                            test_acc)
                        print('New best for {}. Saving model: {}'.format('test_acc', model_save_name))
                        checkpoint = {'global_step': global_step,
                                      'model': model.state_dict(),
                                      'optimizer': optimizer.state_dict(),
                                      'best_test_acc': best_test_acc,
                                      'best_test_steps': best_test_steps
                                      }
                        torch.save(checkpoint, args.checkpoint_directory + '/' + args.model_name + model_save_name)
                    else:
                        pati += 1

                test_writter.flush()

            global_step += 1
    train_writer.close()
    test_writter.close()
    print("train finished!")


if __name__ == '__main__':
    train(args)

