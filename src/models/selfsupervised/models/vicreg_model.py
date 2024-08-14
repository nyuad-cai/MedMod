
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


# Import custom libraries/functions
from src.models.encoders import EHRModel, CXRModel
from src.models.model_utils import load_labels
from src.models.fusion_models import Fusion

gpu_num=os.environ['CUDA_VISIBLE_DEVICES']

# Set cuda device
if gpu_num=='0':
    gpu=[0]
elif gpu_num=='1':
    gpu=[1]
elif gpu_num=='2':
    gpu=[2]
elif gpu_num=='3':
    gpu=[3]
elif gpu_num=='4':
    gpu=[4]
elif gpu_num=='5':
    gpu=[5]
elif gpu_num=='6':
    gpu=[6]
elif gpu_num=='7':
    gpu=[7]
elif gpu_num=='8':
    gpu=[8]
else:
    gpu=['None']
print('Using {} device...'.format(gpu)) 

       
class VICReg(pl.LightningModule):

    def __init__(self, args, train_dl):
                 
        super().__init__()
        assert args.temperature > 0.0, 'The temperature must be a positive float!'
        self.warmup_epochs= 10 #int(0.05*max_epochs) (10 as in SimCLR)
        self.automatic_optimization = False
        
        self.num_train_batches=len(train_dl)
        self.batch_size=args.batch_size
        hidden_dim=args.hidden_dim
        self.args=args
        self.LABEL_COLUMNS = load_labels(args.task, args.labels_set)
        self.task = args.task
        
        # Load the architecture based on args
        self.model = Fusion(args)
  
    
    def configure_optimizers(self):
        # Scaled learning rate in case of multiple GPUs
        if self.args.num_gpu > 1:
            effective_batchsize = self.args.batch_size*self.args.num_gpu
            scaled_lr = self.args.lr*effective_batchsize/self.args.batch_size
        else:
            scaled_lr = self.args.lr 
                    
        # Optimizer
        optimizer = LARS(self.parameters(), lr=scaled_lr, momentum=0.9, weight_decay=self.args.weight_decay)
        
        # Note that the order of the below affects the initial starting learning rate, hence do not change.
        # Main scheduler
        mainscheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 500, verbose=False)
        # Learning rate warmup
        lambda1= lambda epoch : (epoch+1)/self.warmup_epochs
        warmupscheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda1, verbose=False)
                     
        return [optimizer], [mainscheduler, warmupscheduler]

                
    def logging_status(self, mode):
        if mode == 'train':
            on_step=True
            on_epoch=True
        else:
            on_step=False # Report for the sake of naming but it's not useful
            on_epoch=True
        return on_step, on_epoch
    
    def off_diagonal(self,x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    
    def vicreg_loss(self, feats_ehr, feats_img, mode='train'):
        x = feats_ehr
        y = feats_img
        repr_loss = F.mse_loss(x, y)
        
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)

        std_loss_x = torch.mean(F.relu(1 - std_x)) / 2 
        std_loss_y = torch.mean(F.relu(1 - std_y)) / 2
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2
                
        cov_x = (x.T @ x) / (self.args.batch_size - 1)
        cov_y = (y.T @ y) / (self.args.batch_size - 1)
        
        num_features = len(cov_x) #TODO as arg
                
        cov_loss_x = self.off_diagonal(cov_x).pow_(2).sum().div(num_features)
        cov_loss_y = self.off_diagonal(cov_y).pow_(2).sum().div(num_features)
        cov_loss = self.off_diagonal(cov_x).pow_(2).sum().div(num_features) + self.off_diagonal(cov_y).pow_(2).sum().div(num_features)

        loss = (
            self.args.sim_coeff * repr_loss
            + self.args.std_coeff * std_loss
            + self.args.cov_coeff * cov_loss
        )
        on_step, on_epoch = self.logging_status(mode)
        
        return loss, std_loss_x, std_loss_y, cov_loss_x, cov_loss_y, repr_loss

    def training_step(self, batch, batch_idx):
        mode = 'train'
        opt = self.optimizers()
        opt.zero_grad()
        
        # Forward pass for VICReg
        ehr, imgs, y_ehr, y_cxr, seq_lengths, pairs = batch
        ehr = torch.from_numpy(ehr).float()
        ehr = ehr.to(self.device)
        feats_ehr, feats_img = self.model(ehr, seq_lengths, imgs)
        
        # Compute and log vicreg loss
        loss, std_loss_x, std_loss_y, cov_loss_x, cov_loss_y, repr_loss = self.vicreg_loss(feats_ehr, feats_img, mode)
        self.log(mode+'_std_ehr', std_loss_x, on_step=True, on_epoch=True) 
        self.log(mode+'_std_img', std_loss_y, on_step=True, on_epoch=True)
        self.log(mode+'_cov_ehr', cov_loss_x, on_step=True, on_epoch=True) 
        self.log(mode+'_cov_img', cov_loss_y, on_step=True, on_epoch=True)  
        self.log(mode+'_repr_loss', repr_loss, on_step=True, on_epoch=True)  
        self.log(mode+'_loss', loss, on_step=True, on_epoch=True) 

            
        # Backpropagate
        self.manual_backward(loss)
        
        # Optimizer step
        opt.step()
        
        # Learning rate step
        mainscheduler, warmupscheduler = self.lr_schedulers()
        if (self.trainer.is_last_batch) and (self.trainer.current_epoch < self.warmup_epochs-1):
            warmupscheduler.step()
        elif (self.trainer.is_last_batch) and (self.trainer.current_epoch >= self.warmup_epochs-1):
            mainscheduler.step()

        return {'loss': loss, 'feats_ehr': feats_ehr.detach().cpu(), 'feats_img': feats_img.detach().cpu(), 'y_ehr':y_ehr}
      

    
    def validation_step(self, batch, batch_idx):
        mode='val'
        
        # Forward pass for VICReg
        ehr, imgs, y_ehr, y_cxr, seq_lengths, pairs = batch
        ehr = torch.from_numpy(ehr).float()
        ehr = ehr.to(self.device)
        feats_ehr, feats_img = self.model(ehr, seq_lengths, imgs) 
        
        # Compute and log VICReg loss
        loss, std_loss_x, std_loss_y, cov_loss_x, cov_loss_y, repr_loss = self.vicreg_loss(feats_ehr, feats_img, mode)
        self.log(mode+'_std_ehr', std_loss_x, on_step=True, on_epoch=True) 
        self.log(mode+'_std_img', std_loss_y, on_step=True, on_epoch=True)
        self.log(mode+'_cov_ehr', cov_loss_x, on_step=True, on_epoch=True) 
        self.log(mode+'_cov_img', cov_loss_y, on_step=True, on_epoch=True)  
        self.log(mode+'_repr_loss', repr_loss, on_step=True, on_epoch=True)  
        self.log(mode+'_loss', loss, on_step=True, on_epoch=True) 
        
        return {'loss': loss, 'feats_ehr': feats_ehr.detach().cpu(), 'feats_img': feats_img.detach().cpu(), 'y_ehr':y_ehr}
    
        
    def test_step(self, batch, batch_idx):
        mode='test'
        
        # Forward pass for VICReg
        ehr, imgs, y_ehr, y_cxr, seq_lengths, pairs = batch
        ehr = torch.from_numpy(ehr).float()
        ehr = ehr.to(self.device)
            
        # At test time of VICReg, always return all the layer features
        # NOTE: removed this condition
        # if self.args.mode == 'eval':
        feats_ehr_0, feats_ehr_3, feats_img_0, feats_img_3 = self.model(ehr, seq_lengths, imgs) 
    
        # Compute and log VICReg loss
        loss = self.vicreg_loss(feats_ehr_3, feats_img_3, mode)
        self.log(mode+'_loss_epoch', loss, on_step=True, on_epoch=True) 
    
        return {'loss': loss,   'feats_ehr_0': feats_ehr_0.detach().cpu(), 
                                'feats_ehr_3': feats_ehr_3.detach().cpu(), 
                                'feats_img_0': feats_img_0.detach().cpu(), 
                                'feats_img_3': feats_img_3.detach().cpu(), 
                                'y_ehr':y_ehr}       

def count_parameters(model:nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
def train(model, args, train_loader, val_loader, **kwargs): 
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
                         
        
    # For VICReg training
    else:
        print("VICReg training")
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
        early_stop_callback = EarlyStopping(monitor="val_loss_epoch", min_delta=0.001, mode="min", patience=30)
        trainer = pl.Trainer(default_root_dir=os.path.join(model_path),
                             max_epochs=args.epochs, #gpus=gpu,
                             callbacks=[checkpoints, LearningRateMonitor('epoch'), early_stop_callback],
                             logger=logger,  
                             log_every_n_steps=5,  enable_progress_bar=True,
                             num_sanity_val_steps=0,
                            accelerator='gpu', devices=args.num_gpu, strategy=strategy)

    model_parameters = count_parameters(model)
    print("Model parameters: ", model_parameters)
    trainer.fit(model, train_loader, val_loader)
        
    return trainer


# Call this function if doing test without training
def test(model, args, test_loader, **kwargs):
    if 'logger' not in kwargs:
        trainer = pl.Trainer(default_root_dir=os.path.join(args.save_dir))
    else:
        trainer = pl.Trainer(default_root_dir=os.path.join(args.save_dir), logger=logger)
    
    trainer.test(model, test_loader)
    
    return trainer

