# Import libraries
impor t numpy as np
import argparse
import os
import pandas as pd
import neptune.new as neptune
from pathlib import Path

# Import Pytorch 
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# Import custom functions
import parser as par
from datasets.ehr_dataset import get_datasets
from datasets.cxr_dataset import get_cxr_datasets
from datasets.fusion import load_cxr_ehr
from ehr_preprocess import ehr_funcs
from simclr_model import SimCLR, train, test, prepare_data_features


import warnings
warnings.filterwarnings("ignore")

seed = 1002
torch.manual_seed(seed)
np.random.seed(seed)


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

if __name__ == '__main__':
    
    job_number = args.job_number

    parser = par.initiate_parsing()
    args = parser.parse_args()


    path = Path(args.save_dir)
    path.mkdir(parents=True, exist_ok=True)

    discretizer, normalizer = ehr_funcs(args)
    ehr_train, ehr_val, ehr_test = get_datasets(discretizer,normalizer,args)
    cxr_train, cxr_val, cxr_test = get_cxr_datasets(args)

    fusion_train, fusion_val, fusion_test = load_cxr_ehr(args,
                                                        ehr_train_ds=ehr_train,
                                                        ehr_val_ds=ehr_val,
                                                        ehr_test_ds=ehr_test,
                                                        cxr_train_ds=cxr_train,
                                                        cxr_val_ds=cxr_val,
                                                        cxr_test_ds=cxr_test)

    model = SimCLR(args, train_dl)

    train(model, args, train_dl, val_dl,
                logger=neptune_logger,
                load_state_prefix=args.load_state_simclr)
    
    
    test(model, args, test_dl, logger=neptune_logger)

def train(model, args, train_loader, val_loader, **kwargs): 
    filename = args.file_name+'_epoch_{epoch:02d}'
    
    model_path = args.save_dir+'/'+args.file_name
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    
        logger = kwargs['logger']
        checkpoints = ModelCheckpoint(dirpath=model_path,
                                  filename=filename,
                                  save_weights_only=True, 
                                  save_top_k=1,
                                  auto_insert_metric_name=False, 
                                  every_n_epochs=1,             
                                  save_on_train_epoch_end=True)
        if args.num_gpu == 1:
            strategy = None
        else:
            strategy = 'ddp'
       
        early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.001, mode="min", patience=30)

        trainer = pl.Trainer(default_root_dir=os.path.join(model_path),
                             max_epochs=args.epochs,
                             callbacks=[checkpoints, LearningRateMonitor('epoch'), early_stop_callback],
                             logger=logger,  
                             log_every_n_steps=5,  enable_progress_bar=True,
                             num_sanity_val_steps=0,
                            accelerator='gpu', devices=args.num_gpu, strategy=strategy)
                            
    trainer.fit(model, train_loader, val_loader)
        
    return trainer


     
        

    
    
