import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, LayerNorm, to_hetero

# 1. ENCODER
class GNNEncoder(torch.nn.Module):
    def __init__(self, num_nodes, embedding_dim=50, hidden_dim=64, out_dim=50, edge_feature_dim=28, heads=4):
        super().__init__()
        self.embedding = nn.Embedding(num_nodes, embedding_dim)
        
        # GATv2Conv can use edge_dim to process edge features
        self.conv1 = GATv2Conv(
            embedding_dim, hidden_dim, 
            heads=heads, 
            edge_dim=edge_feature_dim,
            add_self_loops=False
        )
        self.ln1 = LayerNorm(hidden_dim * heads)
        self.conv2 = GATv2Conv(
            hidden_dim * heads, out_dim, 
            heads=1, # Output is just 'out_dim'
            edge_dim=edge_feature_dim,
            add_self_loops=False
        )
        self.ln2 = LayerNorm(out_dim)

    def forward(self, x, edge_index, edge_attr=None):
        x = self.embedding(x)
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = self.ln1(x).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = self.ln2(x)
        return x

# 2. DECODER
class EdgeDecoder(torch.nn.Module):
    def __init__(self, in_dim=100, hidden_dim=32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, z, edge_label_index):
        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]
        x = torch.cat([src, dst], dim=1)
        return self.mlp(x).view(-1)

# 3. FULL MODEL
class GNNModel(torch.nn.Module):
    def __init__(self, num_nodes, metadata, embedding_dim=50, hidden_dim=64, out_dim=50, edge_feature_dim=28):
        super().__init__()
        
        self.encoder_base = GNNEncoder(
            num_nodes, embedding_dim, hidden_dim, out_dim, edge_feature_dim=edge_feature_dim
        )
        
        self.encoder_hetero = to_hetero(self.encoder_base, metadata, aggr='mean')
        
        self.decoder = EdgeDecoder(out_dim * 2, 32)

    def encode(self, x_dict, edge_index_dict, edge_attr_dict):
        z_dict = self.encoder_hetero(x_dict, edge_index_dict, edge_attr_dict)
        return z_dict['team_year']