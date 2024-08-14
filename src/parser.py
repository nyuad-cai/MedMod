import argparse

def initiate_parsing():
    parser = argparse.ArgumentParser()
    
    # Task setup
    parser.add_argument('--device', type=str, help='cuda device', default='0')
    parser.add_argument('--num_gpu', type=int, help='number of gpus for training', default=1)
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0001, help='base learning rate simclr pretraining')
    parser.add_argument('--save_dir', type=str, help='Directory where all output files are stored', default='/scratch/se1525/mml-ssl/results')
    parser.add_argument('--labels_set', type=str, default='pheno', help='pheno, radiology')
    parser.add_argument('--task', type=str, default='phenotyping', help='train or eval for in-hospital-mortality or phenotyping')
    parser.add_argument('--data_pairs', type=str, default='paired', help='paired, ehr_only, radiology, joint_ehr')
    parser.add_argument('--mode', type=str, default="train", help='mode: train or eval')  
    parser.add_argument('--tag', type=str, default="simclr train", help='tag for neptune')      
    parser.add_argument('--pretrain_type', type=str, default="simclr", help='type of pretraining')    
    parser.add_argument('--file_name', type=str, default=None, help='prefix of model file name')      
    parser.add_argument('--load_state', type=str, default=None, help='state dir path for simclr model')
    parser.add_argument('--eval_set', type=str, default='val', help='evaluation set: val or test')
    parser.add_argument('--job_number', type=str, default='0', help='slurm job number for jubail')
    parser.add_argument('--eval_epoch', type=int, help='epoch to evaluate for model selection', default=0)
    parser.add_argument('--retrieve_cxr',type=str,default='recent', choices=['recent','all'], help='either to retrieve all cxr or only the most recent')

    
    # EHR setup
    parser.add_argument('--load_state_ehr', type=str, default=None, help='state dir path for uni ehr model')
    parser.add_argument('--num_classes', type=int, default=25, help='number of classes for ehr related tasks')
    parser.add_argument('--rec_dropout', type=float, default=0.0, help="dropout rate for recurrent connections")
    parser.add_argument('--timestep', type=float, default=1.0, help="fixed timestep used in the dataset")
    parser.add_argument('--imputation', type=str, default='previous')
    parser.add_argument('--ehr_data_root', type=str, help='Path to the ehr data', default='/scratch/fs999/shamoutlab/data/mimic-iv-extracted')
    parser.add_argument('--layers', default=1, type=int, help='number of lstm stacked layers')
    parser.add_argument('--dim', type=int, default=256,
                        help='number of hidden units for uni ehr lstm model')

    # CXR setup
    parser.add_argument('--load_state_cxr', type=str, default=None, help='state dir path for uni cxr model')
    parser.add_argument('--cxr_data_root', type=str, help='Path to the cxr data', default='/scratch/fs999/shamoutlab/data/physionet.org/files/mimic-cxr-jpg/2.0.0')
    parser.add_argument('--vision-backbone', default='densenet121', type=str, help='[densenet121, densenet169, densenet201, resnet34]')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',  help='load imagenet pretrained model for cxr')
    parser.add_argument('--layer_after', default=4, type=int, help='apply mmtm/daft module after fourth layer[-1, 0,1,2,3,4] -1 indicates mmtm after every layer')
    parser.add_argument('--vision_num_classes', default=14, type=int, help='number of cxr classes')
    parser.add_argument('--resize', default=256, type=int, help='cxr transform resize')
    parser.add_argument('--crop', default=224, type=int, help='cxr transform crop size')
    parser.add_argument('--dropout', type=float, default=0.0)# TODO: double check
    parser.add_argument('--hidden_dim', type=int, default=128)# TODO: double check



    # SimCLR setup
    parser.add_argument('--load_state_simclr', type=str, default=None, help='state dir path for simclr model')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--transforms_cxr', type=str, default=None, help='set image transforms of simclrv2')
    parser.add_argument('--temperature', type=float, default=0.1, help='simclr temperature')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--finetune', action='store_true',  help='finetune simclr')  
    parser.add_argument('--dataset', type=str, default='evaluation_task', help='type of dataset to work with (all being unrestricted pairs)')
    parser.add_argument('--width', type=int, default=1, help='width of projection module')
    parser.add_argument('--save_features', action='store_true', help='save features after each epoch')
    parser.add_argument('--beta_infonce', action='store_true', help='include time difference in loss computation')


    
    parser.add_argument('--linearclassify', action='store_true',  help='perform linear classification after simclr')  
    parser.add_argument('--load_state_lc', type=str, default=None, help='state dir path for linear class model')
    parser.add_argument('--lr_linearclassify', type=float, default=0.0001, help='learning rate for linear classification')
    parser.add_argument('--epochs_linearclassify', type=int, default=100, help='number of epochs to train for for linear class')
    parser.add_argument('--overwrite_classifier', action='store_true',  help='retrain the logistic regression model and overwrite')    

    
    # Fusion setup
    parser.add_argument('--fusion_type', type=str, default='None', help='train or eval for fusion types [joint, early, uni_cxr, uni_ehr, lstm]')
    parser.add_argument('--data_ratio', type=float, default=1.0, help='percentage of uppaired data samples')
    parser.add_argument('--mmtm_ratio', type=float, default=4, help='mmtm ratio hyperparameter')
    parser.add_argument('--fusion_layer', type=int, default=0, help='fusion layer')



    # Unknown classify later 
    parser.add_argument('--beta_1', type=float, default=0.9,
                        help='beta_1 param for Adam optimizer')
    parser.add_argument('--normalizer_state', type=str, default=None,
                        help='Path to a state file of a normalizer. Leave none if you want to use one of the provided ones.')
    
    
    # Vicreg
    parser.add_argument('--sim_coeff', type=float, default=25, help='vicreg sim coeff')
    parser.add_argument('--std_coeff', type=float, default=25, help='vicreg std coeff')
    parser.add_argument('--cov_coeff', type=float, default=1, help='vicreg cov coeff')
    parser.add_argument('--vicreg', action='store_true', help='vicreg loss computation')

    # ALIGN
    parser.add_argument('--align', action='store_true', help='load align model weights')
   
   # MedCLR
    parser.add_argument('--k', type=float, default=1, help='parameter for medclr loss')

    #For dataloader changes
    parser.add_argument('--tmp_dir', type=str, default=None, help='temporary directory to store data for current node')
   
    return parser