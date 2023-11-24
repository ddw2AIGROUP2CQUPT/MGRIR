# -*- coding: utf-8 -*-
# @Time    : 2023/4/19 19:26
# @Author  : lan
# @File    : model_all2.py
# @Software: PyCharm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool, GENConv, DeepGCNLayer, SAGPooling, \
    TopKPooling
import torch_geometric
from torch.nn import Sequential as Seq
from torch_geometric.nn.norm import BatchNorm, GraphNorm, LayerNorm





import torch
from torch_geometric import nn

import torch.nn.functional as F  # 这个直接在forwrd里面用relu
# torch.nn.RELU() # 这个需要预定义，torch.nn.Linear,torch_geometric.nn.Linear也是 torch.nn.functional.linear需要手动，

from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool, GENConv, DeepGCNLayer, SAGPooling, \
    TopKPooling, BatchNorm, GraphNorm, LayerNorm

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


class Net_chan5p1(torch.nn.Module):
    def __init__(self, in_feats=14, num_heads=8, num_class=10):
        super(Net_chan5p1, self).__init__()

        self.gat1 = GAT(in_feats=in_feats, out_feats=16, num_heads=num_heads) # GAT的std

        self.gcn1 = GCN(in_feats=16,out_feats=32)
        
        self.gat2 = GAT(in_feats=32, out_feats=64, num_heads=num_heads//4) # GAT的std
        
        self.drop = torch.nn.Dropout(0.3)
        
        self.fc = nn.Linear(64, num_class)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = data.batch

        h = self.gat1(x, edge_index,edge_attr)

        h = self.gcn1(h, edge_index)
        
        h = self.gat2(h, edge_index,edge_attr)
        

        h = global_mean_pool(h, batch)
        
        h = self.drop(h)
        y = self.fc(h)

        return y
        

        
class Net_chan5p2(torch.nn.Module):
    def __init__(self, in_feats=14, num_heads=8, num_class=10):
        super(Net_chan5p2, self).__init__()

        self.gat1 = GAT(in_feats=in_feats, out_feats=16, num_heads=num_heads) # GAT的std

        self.gcn1 = GCN(in_feats=16,out_feats=48)
        
        self.gat2 = GAT(in_feats=48, out_feats=128, num_heads=num_heads//4) # GAT的std
        
        self.drop = torch.nn.Dropout(0.3)
        self.fc = nn.Linear(128, num_class)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = data.batch

        h = self.gat1(x, edge_index,edge_attr)

        h = self.gcn1(h, edge_index)
        
        h = self.gat2(h, edge_index,edge_attr)
        

        h = global_mean_pool(h, batch)
        
        
        h = self.drop(h)
        y = self.fc(h)

        return y

#14-24
class Net_chan5p3(torch.nn.Module):
    def __init__(self, in_feats=14, num_heads=8, num_class=10):
        super(Net_chan5p3, self).__init__()

        self.gat1 = GAT(in_feats=in_feats, out_feats=24, num_heads=num_heads) # GAT的std

        self.gcn1 = GCN(in_feats=24,out_feats=32)
        
        self.gat2 = GAT(in_feats=32, out_feats=64, num_heads=num_heads//4) # GAT的std
        
        self.drop = torch.nn.Dropout(0.3)
        self.fc = nn.Linear(64, num_class)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = data.batch

        h = self.gat1(x, edge_index,edge_attr)

        h = self.gcn1(h, edge_index)
        
        h = self.gat2(h, edge_index,edge_attr)
        

        h = global_mean_pool(h, batch)

        h = self.drop(h)
        y = self.fc(h)

        return y
        
class Net_chan5p4(torch.nn.Module):
    def __init__(self, in_feats=14, num_heads=8, num_class=10):
        super(Net_chan5p4, self).__init__()

        self.gat1 = GAT(in_feats=in_feats, out_feats=24, num_heads=num_heads) # GAT的std

        self.gcn1 = GCN(in_feats=24,out_feats=48)
        
        self.gat2 = GAT(in_feats=48, out_feats=128, num_heads=num_heads//4) # GAT的std
        

        self.drop = torch.nn.Dropout(0.3)
        self.fc = nn.Linear(128, num_class)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = data.batch

        h = self.gat1(x, edge_index,edge_attr)

        h = self.gcn1(h, edge_index)
        
        h = self.gat2(h, edge_index,edge_attr)
        

        h = global_mean_pool(h, batch)

        h = self.drop(h)
        y = self.fc(h)

        return y
        
class Net_chan5p5(torch.nn.Module):
    def __init__(self, in_feats=14, num_heads=8, num_class=10):
        super(Net_chan5p5, self).__init__()

        self.gat1 = GAT(in_feats=in_feats, out_feats=24, num_heads=num_heads) # GAT的std

        self.gcn1 = GCN(in_feats=24,out_feats=64)
        
        self.gat2 = GAT(in_feats=64, out_feats=128, num_heads=num_heads//4) # GAT的std
        

        self.drop = torch.nn.Dropout(0.3)
        self.fc = nn.Linear(128, num_class)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = data.batch

        h = self.gat1(x, edge_index,edge_attr)

        h = self.gcn1(h, edge_index)
        
        h = self.gat2(h, edge_index,edge_attr)
        

        h = global_mean_pool(h, batch)

        h = self.drop(h)
        y = self.fc(h)

        return y
        
        
        
class Net_chan5p6(torch.nn.Module):
  def __init__(self, in_feats=14, num_heads=8, num_class=10):
      super(Net_chan5p6, self).__init__()

      self.gat1 = GAT(in_feats=in_feats, out_feats=24, num_heads=num_heads) # GAT的std

      self.gcn1 = GCN(in_feats=24,out_feats=48)
      
      self.gat2 = GAT(in_feats=48, out_feats=96, num_heads=num_heads//4) # GAT的std
      

      self.drop = torch.nn.Dropout(0.3)
      self.fc = nn.Linear(96, num_class)

  def forward(self, data):
      x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
      batch = data.batch

      h = self.gat1(x, edge_index,edge_attr)

      h = self.gcn1(h, edge_index)
      
      h = self.gat2(h, edge_index,edge_attr)
      

      h = global_mean_pool(h, batch)

      h = self.drop(h)
      y = self.fc(h)

      return y

# 16+h12
class Net_chan5p7(torch.nn.Module):
    def __init__(self, in_feats=14, num_heads=12, num_class=10):
        super(Net_chan5p7, self).__init__()

        self.gat1 = GAT(in_feats=in_feats, out_feats=16, num_heads=num_heads) # GAT的std

        self.gcn1 = GCN(in_feats=16,out_feats=32)
        
        self.gat2 = GAT(in_feats=32, out_feats=64, num_heads=num_heads//4) # GAT的std
        

        self.drop = torch.nn.Dropout(0.3)
        self.fc = nn.Linear(64, num_class)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = data.batch

        h = self.gat1(x, edge_index,edge_attr)

        h = self.gcn1(h, edge_index)
        
        h = self.gat2(h, edge_index,edge_attr)
        

        h = global_mean_pool(h, batch)

        h = self.drop(h)
        y = self.fc(h)

        return y

# 24+h12
class Net_chan5p8(torch.nn.Module):
  def __init__(self, in_feats=14, num_heads=12, num_class=10):
      super(Net_chan5p8, self).__init__()

      self.gat1 = GAT(in_feats=in_feats, out_feats=24, num_heads=num_heads) # GAT的std

      self.gcn1 = GCN(in_feats=24,out_feats=48)
      
      self.gat2 = GAT(in_feats=48, out_feats=96, num_heads=num_heads//4) # GAT的std
      

      self.drop = torch.nn.Dropout(0.3)
      self.fc = nn.Linear(96, num_class)

  def forward(self, data):
      x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
      batch = data.batch

      h = self.gat1(x, edge_index,edge_attr)

      h = self.gcn1(h, edge_index)
      
      h = self.gat2(h, edge_index,edge_attr)
      

      h = global_mean_pool(h, batch)

      h = self.drop(h)
      y = self.fc(h)

      return y             

# 32+h12
class Net_chan5p9(torch.nn.Module):
  def __init__(self, in_feats=14, num_heads=12, num_class=10):
      super(Net_chan5p9, self).__init__()

      self.gat1 = GAT(in_feats=in_feats, out_feats=32, num_heads=num_heads) # GAT的std

      self.gcn1 = GCN(in_feats=32,out_feats=64)
      
      self.gat2 = GAT(in_feats=64, out_feats=128, num_heads=num_heads//4) # GAT的std
      

      self.drop = torch.nn.Dropout(0.3)
      self.fc = nn.Linear(128, num_class)

  def forward(self, data):
      x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
      batch = data.batch

      h = self.gat1(x, edge_index,edge_attr)

      h = self.gcn1(h, edge_index)
      
      h = self.gat2(h, edge_index,edge_attr)
      

      h = global_mean_pool(h, batch)

      h = self.drop(h)
      y = self.fc(h)

      return y