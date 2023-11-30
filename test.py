# -*- coding: utf-8 -*-
# @Author  : lan
# @Software: PyCharm
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

import sys

sys.path.append('..')

from models.mnist_model_all2 import *


import numpy as np

from torch.utils.data import Dataset
from torch_geometric.data import Data, DataListLoader  # DataLoader

from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument('--mode', type=str, help=" train or val")
parser.add_argument('--model_name', type=str, help="the model name", default='aa')
parser.add_argument('--gpu', type=str, help='gpu id', default='0,1')

# dataset
parser.add_argument('--num_classes', type=int, help="classification number", default=10)
parser.add_argument('--data_type', type=str, default='cifar10')
parser.add_argument('--data_dir', type=str, help="the path of your train datasets",
                    default='./graph/mnist/mnist_h5/')

parser.add_argument('--num_features', type=int, help='the dim of graph features', default=14)

parser.add_argument('--do_crop', help='do graph augument or not ', action='store_true')
parser.add_argument('--crop_size', type=int, help=' the size of area for the crop operation.', default=224)
parser.add_argument('--test_data_path', type=str, help="the path of your datasets using for online test",
                    default='')
parser.add_argument('do_graphstream', help='update the training data online', action='store_true')
# training
parser.add_argument('--batch_size', type=int, help="batch size", default=128)
parser.add_argument('--total_steps', type=int, help='the total iteration number', default=200000)
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
parser.add_argument('--test_summary_directory', type=str, help='output directory for test summary,'
                                                               'if empty outputs to checkpoint folder', default='')
parser.add_argument('--patience', type=int, help='patience times to adjust lr if test acc can not be better',
                    default=150)

parser.add_argument('--checkpoint_directory', type=str, help='directory to save model', default='checkpoints')

args = parser.parse_args()


print('args', args)


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


import os
import h5py


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


dataset = MyDataset(args.data_dir, split='train')
dataset_test = MyDataset(args.data_dir, split='test')

# DataLoader

train_loader = DataListLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
dataloader_test = DataListLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=0)

print('length of: train_dataset', len(dataset))
print('length of: test_dataset', len(dataset_test))

def online_test(args):
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

    model = Net_chan5p1(in_feats=14, num_heads=12, num_class=10)

    criterion = torch.nn.CrossEntropyLoss()

    base_lr = str2list(args.lr_group, type='float')

    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr.pop(0), weight_decay=args.weight_decay)
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

    print("== Model Initialized")

    print("*" * 10, 'start build dataloader', "*" * 10)

    
    checkpoint = torch.load('./checkpoints/'+args.checkpoint_path,map_location='cpu') # aa is the model name

    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])


    print("train data number:{},test data number:{}".format(len(dataset), len(dataset_test)))


    te_loss_sum = 0
    te_num_correct = 0
    model.eval()
    for _, tr_sample_batched in enumerate(train_loader):
        with torch.no_grad():
            labels_test = [sb.y for sb in tr_sample_batched]
            labels_test = torch.stack(labels_test).cuda()
            outputs = model(tr_sample_batched)
            preds10_score = torch.softmax(outputs, dim=1)
            loss = criterion(outputs, labels_test.long())
            
            te_loss_sum += loss.item()  # * label.size(0)
            targets = labels_test.detach().cpu().numpy()
            preds2 = np.argmax(preds10_score.detach().cpu().numpy(), axis=1)
            corre = (preds2 == targets).sum()  # 计算这个batch正确的个数
            te_num_correct += corre.item()
    test_loss = te_loss_sum / (len(train_loader))  # num / bS
    test_acc = te_num_correct / (len(dataset))  # num
    print(args.model_name, "  的训练loss是%.8f, 训练精度是%.8f" % (test_loss, test_acc*100))



if __name__ == '__main__':
    online_test(args)

