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
import math
# Import other useful libraries
from sklearn.linear_model import LogisticRegression as LR
from sklearn.neural_network import MLPClassifier
import pickle
from flash.core.optimizers import LARS
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import metrics
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
from copy import deepcopy
from tqdm import tqdm
import pandas as pd

# Import custom libraries/functions
from encoders import LSTM, CXRModels
import load_tasks as tasks
from fusion_models import Fusion
from utils import get_model_performance, get_model_performance_los, get_bin_custom, CustomBins, mean_absolute_percentage_error

       
class SimCLR(pl.LightningModule):

    def __init__(self, args, train_dl):

        super().__init__()
        assert args.temperature > 0.0, 'The temperature must be a positive float!'
        self.warmup_epochs= 10 
        self.automatic_optimization = False
        
        self.num_train_batches=len(train_dl)
        self.batch_size=args.batch_size
        hidden_dim=args.hidden_dim
        self.args=args
        self.LABEL_COLUMNS = tasks.load_labels(args.task, args.labels_set)
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


    
    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        opt.zero_grad()
        mode = 'train'

        # Forward pass for SimCLR
        ehr, imgs, y_ehr, y_cxr, seq_lengths, pairs = batch
        ehr = torch.from_numpy(ehr).float()
        ehr = ehr.to(self.device)
        feats_ehr, feats_img = self.model(ehr, seq_lengths, imgs)
        loss = self.info_nce_loss(feats_ehr, feats_img, mode)
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
        # Forward pass for SimCLR
        ehr, imgs, y_ehr, y_cxr, seq_lengths, pairs = batch
        ehr = torch.from_numpy(ehr).float()
        ehr = ehr.to(self.device)
        feats_ehr, feats_img = self.model(ehr, seq_lengths, imgs) 
        
        # Compute and log infoNCE loss
        loss = self.info_nce_loss(feats_ehr, feats_img, mode)
        self.log(mode+'_loss_epoch', loss, on_step=False, on_epoch=True) #, logger=True)
        
        return {'loss': loss, 'feats_ehr': feats_ehr.detach().cpu(), 'feats_img': feats_img.detach().cpu(), 'y_ehr':y_ehr}
        
        
        
    def test_step(self, batch, batch_idx):
        mode='test'
        # Forward pass for SimCLR

        ehr, imgs, y_ehr, y_cxr, seq_lengths, pairs = batch
        ehr = torch.from_numpy(ehr).float()
        ehr = ehr.to(self.device)
            
        # At test time of SIMCLR, always return all the layer features
        if self.args.mode == 'eval':
            feats_ehr_0, feats_ehr_3, feats_img_0, feats_img_3 = self.model(ehr, seq_lengths, imgs) 
        
            # Compute and log infoNCE loss
            if self.args.beta_infonce == True:
                k = self.args.k
                loss = self.modified_info_nce_loss(feats_ehr_3, feats_img_3, time_diff, k,  mode)
            else:
                loss = self.info_nce_loss(feats_ehr_3, feats_img_3, mode)
            self.log(mode+'_loss_epoch', loss, on_step=False, on_epoch=True) #, logger=True)
        
            return {'loss': loss,   'feats_ehr_0': feats_ehr_0.detach().cpu(), 
                                    'feats_ehr_3': feats_ehr_3.detach().cpu(), 
                                    'feats_img_0': feats_img_0.detach().cpu(), 
                                    'feats_img_3': feats_img_3.detach().cpu(), 
                                    'y_ehr':y_ehr}        

                
    
    def process_features(self, outputs, mode):
        y = []
        if self.args.mode=='eval':
            feats_ehr_0=[]
            feats_ehr_3=[]
            feats_img_0=[]
            feats_img_3=[]
        else:
            feats_ehr = []
            feats_img = []
        # Iterate through batches and append
        i=0
        for output in outputs:
            if i ==0:
                if self.args.fusion_type!='None':
                    preds = output['preds'].detach().cpu()
                elif self.args.mode == 'eval':
                    feats_ehr_0 = output['feats_ehr_0'].detach().cpu()
                    feats_ehr_3 = output['feats_ehr_3'].detach().cpu()
                    feats_img_0 = output['feats_img_0'].detach().cpu()
                    feats_img_3 = output['feats_img_3'].detach().cpu()
                else: 
                    feats_ehr = output['feats_ehr'].detach().cpu()
                    feats_img = output['feats_img'].detach().cpu()
                y = output['y'].tolist()
                
            else:
                if self.args.fusion_type!='None':
                    preds = torch.cat((preds, output['preds'].detach().cpu()))
                elif self.args.mode == 'eval':
                    feats_ehr_0 = torch.cat((feats_ehr_0, output['feats_ehr_0'].detach().cpu()))
                    feats_ehr_3 = torch.cat((feats_ehr_3, output['feats_ehr_3'].detach().cpu()))
                    feats_img_0 = torch.cat((feats_img_0, output['feats_img_0'].detach().cpu()))
                    feats_img_3 = torch.cat((feats_img_3, output['feats_img_3'].detach().cpu()))
                else:
                    feats_ehr = torch.cat((feats_ehr, output['feats_ehr'].detach().cpu()))
                    feats_img = torch.cat((feats_img, output['feats_img'].detach().cpu()))
                y.extend(output['y'].tolist())
            i+=1
        if self.args.fusion_type!='None':
            return y, preds
        elif self.args.mode=='eval':
            return feats_ehr_0, feats_ehr_3, feats_img_0, feats_img_3, y
        else:
            return feats_ehr, feats_img, y
    
    def save_features(self, x, descrip, mode):
        model_path = self.args.save_dir+'/simclr_lr/'+self.args.file_name
        if not os.path.exists(model_path):
          os.makedirs(model_path)
        
        torch.save(x, model_path+'/{}_{}_epoch_{}.pt'.format(mode, descrip, self.current_epoch))
    
    def training_epoch_end(self, outputs):
        mode='train'
        feats_ehr, feats_img, y = self.process_features(outputs, mode)
        self.save_features(feats_ehr, 'feats_ehr', mode)
        self.save_features(feats_img, 'feats_img', mode)      
        self.save_features(y, 'y', mode)  
       
    
        
    def validation_epoch_end(self, outputs):
        mode='val'
        feats_ehr, feats_img, y = self.process_features(outputs, mode)
        self.save_features(feats_ehr, 'feats_ehr', mode)
        self.save_features(feats_img, 'feats_img', mode)      
        self.save_features(y, 'y', mode)
       
            

    def test_epoch_end(self, outputs):
        mode = self.args.eval_set
        feats_ehr_0, feats_ehr_3, feats_img_0, feats_img_3, y = self.process_features(outputs, mode)
        self.save_features(feats_ehr_0, 'feats_ehr_0', mode)
        self.save_features(feats_ehr_3, 'feats_ehr_3', mode)
        self.save_features(feats_img_0, 'feats_img_0', mode)
        self.save_features(feats_img_3, 'feats_img_3', mode)      
        self.save_features(y, 'y', mode)

               
            
    def calculate_auroc_epoch(self, outputs, mode):
        labels = []
        predictions = []
        auroc_label={}
        outputs=outputs[self.args.fusion_type].squeeze()
        
        for output in outputs:
            for out_labels in output["labels"].detach().cpu():
                labels.append(out_labels)
            for out_predictions in output["predictions"].detach().cpu():
                predictions.append(out_predictions)

        labels = torch.stack(labels).int()
        predictions = torch.stack(predictions)
        for i, name in enumerate(self.LABEL_COLUMNS):
            class_roc_auc = roc_auc_score(labels[:, i], predictions[:, i])
            auroc_label[name]=class_roc_auc
            
        auroc = roc_auc_score(labels, predictions)
        auprc = average_precision_score(labels, predictions)
        
        return auroc, auprc
       
    
def return_model_version(trainer):
    filename = trainer.checkpoint_callback.filename
    best_model_path = trainer.checkpoint_callback.best_model_path
    return  filename+best_model_path.split('.ckpt')[0].split(filename)[1]

def count_parameters(model:nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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
