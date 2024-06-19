
import random

import torchvision
import torch
import pytorch_lightning as pl
from torch import nn
import torch.nn.functional as F
    

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
        self.log("train_loss", loss, on_epoch= True,on_step=True , logger=True, prog_bar=True)
        
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