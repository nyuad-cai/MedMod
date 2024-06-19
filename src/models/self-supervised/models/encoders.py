# Import Pytorch 
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import torchvision


class EHRModel(nn.Module):

    def __init__(self, 
                 hidden_dim: int =256, 
                 input_dim: int =76,  
                 batch_first: bool = True, 
                 dropout: float = 0.0, 
                 layers: int = 1,
                 projection_dim: int = 512):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layers = layers
        for layer in range(layers):
            setattr(self, f'layer{layer}', nn.LSTM(
                input_dim, hidden_dim,
                batch_first=batch_first,
                dropout = dropout)
            )
            input_dim = hidden_dim

        self.do = nn.Dropout(dropout)
        self.feats_dim = hidden_dim
        self.projection_layer = nn.Linear(hidden_dim, projection_dim)
        self.initialize_weights()

    def initialize_weights(self):
        for model in self.modules():

            if type(model) in [nn.Linear]:
                nn.init.xavier_uniform_(model.weight)
                nn.init.zeros_(model.bias)
            elif type(model) in [nn.LSTM, nn.RNN, nn.GRU]:
                nn.init.orthogonal_(model.weight_hh_l0)
                nn.init.xavier_uniform_(model.weight_ih_l0)
                nn.init.zeros_(model.bias_hh_l0)
                nn.init.zeros_(model.bias_ih_l0)

    def forward(self, x, seq_lengths):
        x = torch.nn.utils.rnn.pack_padded_sequence(x, seq_lengths, batch_first=True, enforce_sorted=False)
        for layer in range(self.layers):
             x, (ht, _) = getattr(self, f'layer{layer}')(x)
        feats = ht.squeeze()
        out = self.do(feats)
        out = self.projection_layer(out)
        return out
    
class CXRModel(nn.Module):

    def __init__(self,
                 backbone: str = 'resnet34',
                 projection_dim: int = 512):
        super().__init__()
        
        self.vision_backbone = getattr(torchvision.models, backbone)(pretrained=False)
        self.vision_backbone.fc = nn.Linear(self.vision_backbone.fc.in_features,projection_dim)



    def forward(self, x: torch.Tensor):
        visual_feats = self.vision_backbone(x)
        return  visual_feats