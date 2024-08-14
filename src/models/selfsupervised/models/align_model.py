# Import Pytorch 
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, Callback, TQDMProgressBar
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import loggers as pl_loggers
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import torchvision

# Import other useful libraries
from sklearn.linear_model import LogisticRegression as LR
from sklearn.neural_network import MLPClassifier
from flash.core.optimizers import LARS
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
import numpy as np
from copy import deepcopy
from tqdm import tqdm
import pandas as pd
import math
import random

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

    

class ALIGN(nn.Module):
    def __init__(self,
                 hidden_dim: int =256, 
                 input_dim: int =76, 
                 batch_first: bool = True, 
                 dropout: float = 0.0, 
                 layers: int = 1,
                 backbone: str = 'resnet34',
                 projection_dim: int = 512):
        super().__init__()
        
        self.cxr_encoder = CXRModel(backbone=backbone,
                                     projection_dim=projection_dim)
        
        self.ehr_encoder = EHRModel(hidden_dim=hidden_dim,
                                    input_dim=input_dim,
                                    batch_first=batch_first,
                                    dropout=dropout,
                                    layers=layers,
                                    projection_dim=projection_dim)
        
    def forward(self,
               cxr: torch.Tensor,
               ehr: torch.Tensor,
               seq_lengths: list):
        
        cxr_projections = self.cxr_encoder(cxr)
        ehr_projections = self.ehr_encoder(ehr,seq_lengths)
        
        return {'cxr': cxr_projections, 
                'ehr': ehr_projections}
    

class ALIGNTrainer(pl.LightningModule):
    def __init__(self,

                 hidden_dim: int =256, 
                 input_dim: int =76, 
                 batch_first: bool = True, 
                 dropout: float = 0.0, 
                 layers: int = 1,
                 backbone: str = 'resnet34',
                 projection_dim: int = 512,
                 temperature: float = 0.07,
                 lr: float = 0.0001,
                 wd=0.001,
                 max_epochs: int = 100):
        super().__init__()


        self.model = ALIGN(hidden_dim=hidden_dim,
                          input_dim=input_dim,
                          batch_first=batch_first,
                          dropout=dropout,
                          layers=layers,
                          backbone=backbone,
                          projection_dim=projection_dim)
        


        self.criterion = ContrastiveLoss(temperature=temperature)        
        
        
        self.lr = lr
        self.wd = wd
        self.max_epochs = max_epochs
        
    
        
    def training_step(self, batch, batch_idx):
        
        ehr, cxr,_ , _, seq_lengths, _ = batch
        ehr,seq_lengths = self._swap(ehr,seq_lengths)
        
        embeddings = self.model(cxr.cuda(),ehr.cuda(),seq_lengths.to('cpu'))
        
        loss = self.criterion(embeddings['cxr'], embeddings['ehr']) 
        self.log("train_loss", loss, on_epoch= True,on_step=True)
        
        return loss



    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(self.parameters(), 
                                      lr=self.lr,
                                      weight_decay=self.wd)
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                               eta_min=0.0,
                                                               T_max=self.max_epochs)
        return {'optimizer': optimizer,
               'lr_scheduler': scheduler
               }
    
    def _swap(self,ehr,seqs):
    
        ehr = torch.tensor(ehr,dtype= torch.float32)
        seqs = torch.tensor(seqs, dtype= torch.float32)
        b_size = ehr.shape[0]
    
        # number of samples to sap
        count = random.randint(int(0.16*b_size),int(0.2*b_size))
    
        # first slice limits retrieval 
        group1_start = random.randint(0,int(0.4*b_size))
        group1_end = group1_start + count
        ehr1 = torch.clone(ehr[group1_start:group1_end])
        seqs1 = torch.clone(seqs[group1_start:group1_end])
    
        # second slice limits retrieval
        group2_start = random.randint(int(0.6*b_size),int(0.8*b_size))
        group2_end = group2_start + count
        ehr2 = torch.clone(ehr[group2_start:group2_end])
        seqs2 = torch.clone(seqs[group2_start:group2_end])
    
        # perform swapping
        ehr[group1_start:group1_end] = ehr2
        seqs[group1_start:group1_end] = seqs2
    
        ehr[group2_start:group2_end] = ehr1
        seqs[group2_start:group2_end] = seqs1
    
        return ehr, seqs


class ContrastiveLoss(nn.Module):
    def __init__(self,
                temperature: float =0.07):
        
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(temperature))


    def forward(self, cxr_feats, ehr_feats):

        cos_sim = F.cosine_similarity(cxr_feats[:,None,:], ehr_feats[None,:,:], dim=-1)

        cos_sim = cos_sim / self.temperature
        self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool,  device=cos_sim.device)

        cos_sim_negative = torch.clone(cos_sim)
        cos_sim_negative.masked_fill_(self_mask, -9e15)
        
        # Compute based on img->ehr
        nll_1 = cos_sim[self_mask] - torch.logsumexp(cos_sim_negative, dim=1)
        
        # Compute based on ehr->img
        nll_2 = cos_sim[self_mask] - torch.logsumexp(cos_sim_negative, dim=0) 

        # Total loss 
        loss = -(nll_1 + nll_2).mean()
                     
        return loss


def count_parameters(model:nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
def train(model, args, train_loader, **kwargs): 
    filename = args.file_name+'_epoch_{epoch:02d}'
    model_path = args.save_dir+'/'+args.file_name
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    
    # For model selection
    if 'logger' not in kwargs:
        print("Model selection")
        checkpoint_callback = ModelCheckpoint(monitor="val_auroc", mode='max',
                                              filename=args.load_state+'_{epoch:02d}',
                                              dirpath=args.save_dir,
                                              save_top_k=1,
                                              every_n_epochs=1,
                                              save_on_train_epoch_end=True
                                              )
        
        trainer = pl.Trainer(default_root_dir=os.path.join(model_path),
                         accelerator="auto",
                         max_epochs=args.epochs, gpus=gpu,
                         callbacks=[checkpoint_callback],
                         enable_progress_bar=False,
                         num_sanity_val_steps=0)
                         
        
    # For ALIGN training
    else:
        print("ALIGN training")
        logger = kwargs['logger']
        checkpoints = ModelCheckpoint(#monitor="val_auroc_epoch", mode='max',
                                    dirpath=model_path,
                                  filename=filename,
                                  save_weights_only=True, 
                                  save_top_k=-1,
                                  auto_insert_metric_name=False, 
                                  every_n_epochs=1,             
                                  save_on_train_epoch_end=True)
        if args.num_gpu == 1:
            strategy = None
        else:
            strategy = 'ddp' 
        early_stop_callback = EarlyStopping(monitor="train_loss", min_delta=0.001, mode="min", patience=30)
        trainer = pl.Trainer(default_root_dir=os.path.join(model_path),
                             max_epochs=args.epochs, 
                             callbacks=[checkpoints, LearningRateMonitor('epoch'), early_stop_callback],
                             logger=logger,  
                             log_every_n_steps=5,  enable_progress_bar=True,
                             num_sanity_val_steps=0,
                            accelerator='gpu', devices=args.num_gpu, strategy=strategy)

    model_parameters = count_parameters(model)
    print("Model parameters: ", model_parameters)
    trainer.fit(model, train_loader)
        
    return trainer


# Call this function if doing test without training
def test(model, args, test_loader, **kwargs):
    if 'logger' not in kwargs:
        trainer = pl.Trainer(default_root_dir=os.path.join(args.save_dir))
    else:
        trainer = pl.Trainer(default_root_dir=os.path.join(args.save_dir), logger=logger)
    
    trainer.test(model, test_loader)
    
    return trainer