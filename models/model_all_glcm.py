# -*- coding: utf-8 -*-
# @Author  : lan
# @Software: PyCharm
import torch

import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool, GENConv, DeepGCNLayer, SAGPooling, \
    TopKPooling
import torch_geometric
from torch.nn import Sequential as Seq
from torch_geometric.nn.norm import BatchNorm, GraphNorm, LayerNorm
from torch_geometric import nn
import numpy as np

class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        
        self.layers = torch.nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.layers(x)

class GCN_block(torch.nn.Module):
    def __init__(self, input_dims, output_dims, head_nums, do_linear=True, linear_outdims=None):
        super(GCN_block, self).__init__()

        self.do_linear = do_linear
        self.conv0 = GATConv(input_dims, output_dims, heads=head_nums, edge_dim=2)  # edge_dim=2
        self.BN0 = BatchNorm(output_dims * head_nums)
        # self.BN0 = LayerNorm(output_dims * head_nums)
        self.relu = torch.nn.ReLU()
        if self.do_linear:
            self.linear = torch.nn.Linear(output_dims * head_nums, linear_outdims)
            # self.BN1 = LayerNorm(linear_outdims)
            self.BN1 = BatchNorm(linear_outdims)

    def forward(self, x, adj, edge_attr):

        x = self.conv0(x, adj, edge_attr=edge_attr)
        x = self.BN0(x)
        x = self.relu(x)

        if self.do_linear:
            x = self.linear(x)
            x = self.BN1(x)
            x = self.relu(x)

        return x

class GCN(torch.nn.Module):
    def __init__(self, in_feats, out_feats):  # hidden_size,
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_feats, out_feats)
        self.batch_norm1 = BatchNorm(out_feats)
        self.res_link = nn.Linear(in_feats, out_feats)

    def forward(self, x, edge_index):
        h = x

        h = self.batch_norm1(F.relu(self.conv1(h, edge_index)))

        res = self.res_link(x)

        return h + res

class GAT(torch.nn.Module):
    def __init__(self, in_feats, out_feats, num_heads):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_feats, out_feats, heads=num_heads)
        self.BatchNorm1 = BatchNorm(out_feats * num_heads)
        self.conv_linear1 = nn.Linear(out_feats * num_heads, out_feats)
        self.BatchNorm2 = BatchNorm(out_feats)

    def forward(self, x, edge_index, edge_attr):
        h = x

        h = self.conv1(h, edge_index, edge_attr)

        h = self.BatchNorm1(h)
        h = F.relu(h)
        h = self.conv_linear1(h)
        h = self.BatchNorm2(h)
        h_gat = F.relu(h)

        return h_gat

# The parameters of the network are 40922
class Net_chan5p1(torch.nn.Module):
    def __init__(self, in_feats=14, num_heads=12, num_class=10):
        super(Net_chan5p1, self).__init__()

        self.mlp = MLP(96, 64, 32)

        self.gat1 = GAT(in_feats=in_feats, out_feats=16, num_heads=num_heads)  # GAT的std

        self.gcn1 = GCN(in_feats=16, out_feats=32)

        self.gat2 = GAT(in_feats=32, out_feats=64, num_heads=num_heads // 4)  # GAT的std

        self.drop = torch.nn.Dropout(0.3)
        self.fc = nn.Linear(int(64 + 32), num_class)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = data.batch

        unique_values, counts = torch.unique(batch, return_counts=True)
        num_graphs = unique_values.numel()  # 子图个数

        # print(batch.shape) # torch.Size([8428])

        glcm = data.glcm.reshape(num_graphs, -1)
        # print('000000000000000',glcm.shape) # 32,96
        glcm = self.mlp(glcm)
        # print('x---------------',x.shape)
        # print('edge_index',edge_index.shape)
        # print('edge_attr',edge_attr.shape)

        h = self.gat1(x, edge_index, edge_attr)

        h = self.gcn1(h, edge_index)

        h = self.gat2(h, edge_index, edge_attr)

        h = global_mean_pool(h, batch)
        h = self.drop(h)

        x = torch.cat([h, glcm], dim=1)

        y = self.fc(x)

        return y

# The parameters of the network are 71890
class Net_chan5p2(torch.nn.Module):
    def __init__(self, in_feats=14, num_heads=12, num_class=10):
        super(Net_chan5p2, self).__init__()
        
        self.mlp = MLP(96, 64, 32)

        self.gat1 = GAT(in_feats=in_feats, out_feats=24, num_heads=num_heads) # GAT的std

        self.gcn1 = GCN(in_feats=24,out_feats=48)
      
        self.gat2 = GAT(in_feats=48, out_feats=96, num_heads=num_heads//4) # GAT的std
        

        self.drop = torch.nn.Dropout(0.3)
        self.fc = nn.Linear(int(96+32), num_class)


    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = data.batch
        
        unique_values, counts = torch.unique(batch, return_counts=True)
        num_graphs = unique_values.numel() # 子图个数
        
        # print(batch.shape) # torch.Size([8428])
        
        glcm = data.glcm.reshape(num_graphs, -1) 
        # print('000000000000000',glcm.shape) # 32,96
        glcm = self.mlp(glcm)
        # print('x---------------',x.shape)
        # print('edge_index',edge_index.shape)
        # print('edge_attr',edge_attr.shape)

        h = self.gat1(x, edge_index,edge_attr)

        h = self.gcn1(h, edge_index)
        
        h = self.gat2(h, edge_index,edge_attr)
        

        h = global_mean_pool(h, batch)
        
        h = self.drop(h)
        
        x = torch.cat([h, glcm], dim=1)

        y = self.fc(x)

        return y

# The parameters of the network are 114122
class Net_chan5p3(torch.nn.Module):
    def __init__(self, in_feats=14, num_heads=12, num_class=10):
        super(Net_chan5p3, self).__init__()
        
        self.mlp = MLP(96, 64, 32)

        self.gat1 = GAT(in_feats=in_feats, out_feats=32, num_heads=num_heads) # GAT的std

        self.gcn1 = GCN(in_feats=32,out_feats=64)
      
        self.gat2 = GAT(in_feats=64, out_feats=128, num_heads=num_heads//4) # GAT的std
        

        self.drop = torch.nn.Dropout(0.3)
        self.fc = nn.Linear(int(128+32), num_class)


    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = data.batch
        
        unique_values, counts = torch.unique(batch, return_counts=True)
        num_graphs = unique_values.numel() # 子图个数
        
        # print(batch.shape) # torch.Size([8428])
        
        glcm = data.glcm.reshape(num_graphs, -1) 
        # print('000000000000000',glcm.shape) # 32,96
        glcm = self.mlp(glcm)
        # print('x---------------',x.shape)
        # print('edge_index',edge_index.shape)
        # print('edge_attr',edge_attr.shape)

        h = self.gat1(x, edge_index,edge_attr)

        h = self.gcn1(h, edge_index)
        
        h = self.gat2(h, edge_index,edge_attr)
        

        h = global_mean_pool(h, batch)
        
        h = self.drop(h)
        
        x = torch.cat([h, glcm], dim=1)

        y = self.fc(x)

        return y

# The parameters of the network are 53458
class Net_chan5p4(torch.nn.Module):
    def __init__(self, in_feats=14, num_heads=8, num_class=10):
        super(Net_chan5p4, self).__init__()

        self.mlp = MLP(96, 64, 32)

        self.gat1 = GAT(in_feats=in_feats, out_feats=24, num_heads=num_heads)  # GAT的std

        self.gcn1 = GCN(in_feats=24, out_feats=48)

        self.gat2 = GAT(in_feats=48, out_feats=96, num_heads=num_heads // 4)  # GAT的std

        self.drop = torch.nn.Dropout(0.3)
        self.fc = nn.Linear(int(96 + 32), num_class)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = data.batch

        unique_values, counts = torch.unique(batch, return_counts=True)
        num_graphs = unique_values.numel()  # 子图个数

        # print(batch.shape) # torch.Size([8428])

        glcm = data.glcm.reshape(num_graphs, -1)
        # print('000000000000000',glcm.shape) # 32,96
        glcm = self.mlp(glcm)
        # print('x---------------',x.shape)
        # print('edge_index',edge_index.shape)
        # print('edge_attr',edge_attr.shape)

        h = self.gat1(x, edge_index, edge_attr)

        h = self.gcn1(h, edge_index)

        h = self.gat2(h, edge_index, edge_attr)

        h = global_mean_pool(h, batch)
        h = self.drop(h)

        x = torch.cat([h, glcm], dim=1)

        y = self.fc(x)

        return y
