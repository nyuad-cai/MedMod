import os
import sys
import pytorch_lightning as pl

from src.models import ALIGNTrainer
from src.parser import initiate_parsing
from src.fusion import load_cxr_ehr
from src.ehr_dataset import get_datasets
from pytorch_lightning.loggers import  CSVLogger
from src.cxr_dataset import get_cxr_datasets
from src.preprocessing import Discretizer, Normalizer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


if __name__ == '__main__':

    parser = initiate_parsing()
    args = parser.parse_args()


    model = ALIGNTrainer(hidden_dim=256,
                        input_dim=76,
                        batch_first=True,
                        dropout=0,
                        layers=1,
                        backbone='resnet34',
                        projection_dim=512,
                        temperature=0.07,
                        lr=args.lr,
                        wd=1e-4,
                        max_epochs=100
                        )

    print(args.lr)
    discretizer = Discretizer()
    normalizer = Normalizer()
    normalizer.load_params('../ph_ts0.8.input_str:previous.start_time:zero.normalizer')

    ehr_train, ehr_val, ehr_test = get_datasets(discretizer,normalizer,args)
    cxr_train, cxr_val, cxr_test = get_cxr_datasets(args)

    fusion_train, fusion_val, fusion_test = load_cxr_ehr(args,
                                                        ehr_train_ds=ehr_train,
                                                        ehr_val_ds=ehr_val,
                                                        ehr_test_ds=ehr_test,
                                                        cxr_train_ds=cxr_train,
                                                        cxr_val_ds=cxr_val,
                                                        cxr_test_ds=cxr_test)


    checkpoint_callback = ModelCheckpoint(monitor='train_loss', 
                                        mode='min',
                                        every_n_epochs=1,
                                        save_top_k=-1,
                                        )

    early_stop = EarlyStopping(monitor='train_loss', 
                            mode='min', 
                            patience=30)

    csv_logger = CSVLogger(save_dir= os.path.join('default'),version=os.getenv('RUN'),
                        flush_logs_every_n_steps=1)

    trainer = pl.Trainer(gpus=1,
                        logger=csv_logger, 
                        log_every_n_steps=1,
                        max_epochs=args.epochs,
                        callbacks=[checkpoint_callback,early_stop],
                        default_root_dir='.'
                        )

    trainer.fit(model=model, train_dataloaders=fusion_train)