import os
import glob
import torch
import numpy as np
import pandas as pd
import torchvision.transforms as transforms

from random import choice
from torch.utils.data import DataLoader
from .datasets import MIMICCXR, EHRdataset, MIMIC_CXR_EHR

# chunk-wise reading
def read_chunk(reader, chunk_size):
    data = {}
    for i in range(chunk_size):
        ret = reader.read_next()
        for k, v in ret.items():
            if k not in data:
                data[k] = []
            data[k].append(v)
    data["header"] = data["header"][0]
    return data


# CXR dataset utils
def get_transforms(args):
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    
    train_transforms = []
    train_transforms.append(transforms.Resize(args.resize))
    train_transforms.append(transforms.RandomHorizontalFlip())
    train_transforms.append(transforms.RandomAffine(degrees=45, scale=(.85, 1.15), shear=0, translate=(0.15, 0.15)))
    train_transforms.append(transforms.CenterCrop(224))
    train_transforms.append(transforms.ToTensor())
     

    test_transforms = []
    test_transforms.append(transforms.Resize(args.resize))
    test_transforms.append(transforms.CenterCrop(224))
    test_transforms.append(transforms.ToTensor())

    return train_transforms, test_transforms



class RandomCrop(object):
    "Randomly crop an image"
    
    def __call__(self, sample):
        resize = 256
        random_crop_size = int(np.random.uniform(0.6*resize,resize,1))
        sample=transforms.RandomCrop(random_crop_size)(sample)
        return sample
    
    

class RandomColorDistortion(object):
    "Apply random color distortions to the image"
    
    def __call__(self, sample):
        resize=256

        # Random color distortion
        strength = 1.0 # 1.0 imagenet setting and CIFAR uses 0.5
        brightness = 0.8 * strength 
        contrast = 0.8 * strength
        saturation = 0.8 * strength
        hue = 0.2 * strength
        prob = np.random.uniform(0,1,1) 
        if prob < 0.8:
            sample=transforms.ColorJitter(brightness, contrast, saturation, hue)(sample)

        # Random Grayscale
        sample=transforms.RandomGrayscale(p=0.2)(sample)
        return sample 
    


def get_transforms_simclr(args):
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    
    train_transforms = []
    # Resize all images to same size, then randomly crop and resize again
    train_transforms.append(transforms.Resize([args.resize, args.resize]))
    # Random affine
    train_transforms.append(transforms.RandomAffine(degrees=(-45, 45), translate=(0.1,0.1), scale=(0.7, 1.5), shear=(-25, 25)))
    # Random crop
    train_transforms.append(RandomCrop())
    # Resize again
    # train_transforms.append(transforms.Resize([args.resize, args.resize], interpolation=3))
    train_transforms.append(transforms.Resize([224, 224], interpolation=3))
    # Random horizontal flip 
    train_transforms.append(transforms.RandomHorizontalFlip())
    # Random color distortions
    train_transforms.append(RandomColorDistortion())
    # Convert to tensor
    train_transforms.append(transforms.ToTensor())
    
    test_transforms = []
    # Resize all images to same size, then center crop and resize again
    test_transforms.append(transforms.Resize([args.resize, args.resize]))
    crop_proportion=0.875
    test_transforms.append(transforms.CenterCrop([int(0.875*args.resize), int(0.875*args.resize)]))
    # test_transforms.append(transforms.Resize([args.resize, args.resize], interpolation=3))
    test_transforms.append(transforms.Resize([224, 224], interpolation=3))
    #Convert to tensor
    test_transforms.append(transforms.ToTensor())

    return train_transforms, test_transforms




def get_cxr_datasets(args):
    if args.transforms_cxr=='simclrv2':
        train_transforms, test_transforms = get_transforms_simclr(args)
    else:
        train_transforms, test_transforms = get_transforms(args)

    data_dir = args.cxr_data_root
    filepath = f'{data_dir}/new_paths.npy'
    if os.path.exists(filepath):
        paths = np.load(filepath)
    else:
        paths = glob.glob(f'{args.tmp_dir}/resized/**/*.jpg', recursive = True)
        np.save(filepath, paths)
    dataset_train = MIMICCXR(paths, args, split='train', transform=transforms.Compose(train_transforms))
    dataset_validate = MIMICCXR(paths, args, split='validate', transform=transforms.Compose(test_transforms),)
    dataset_test = MIMICCXR(paths, args, split='test', transform=transforms.Compose(test_transforms),)

    return dataset_train, dataset_validate, dataset_test




############################################################################

# EHR utils
def get_datasets(discretizer, normalizer, args):
  
    transform = None
    train_ds = EHRdataset(args, discretizer, normalizer, f'{args.ehr_data_root}/{args.task}/train_listfile.csv', os.path.join(args.ehr_data_root, f'{args.task}/train'), transforms=transform)
    val_ds = EHRdataset(args, discretizer, normalizer, f'{args.ehr_data_root}/{args.task}/val_listfile.csv', os.path.join(args.ehr_data_root, f'{args.task}/train'), transforms = transform)
    test_ds = EHRdataset(args, discretizer, normalizer, f'{args.ehr_data_root}/{args.task}/test_listfile.csv', os.path.join(args.ehr_data_root, f'{args.task}/test'), transforms = transform)
    return train_ds, val_ds, test_ds


def get_data_loader(discretizer, normalizer, dataset_dir, batch_size):
    train_ds, val_ds, test_ds = get_datasets(discretizer, normalizer, dataset_dir)
    train_dl = DataLoader(train_ds, batch_size, shuffle=False, collate_fn=my_collate_ehr, pin_memory=True, num_workers=16)
    val_dl = DataLoader(val_ds, batch_size, shuffle=False, collate_fn=my_collate_ehr, pin_memory=True, num_workers=16)
    return train_dl, val_dl
        
def my_collate_ehr(batch):
    x = [item[0] for item in batch]
    x, seq_length = pad_zeros(x)
    targets = np.array([item[1] for item in batch])
    return [x, targets, seq_length]

def pad_zeros(arr, min_length=None):

    dtype = arr[0].dtype
    seq_length = [x.shape[0] for x in arr]
    max_len = max(seq_length)
    ret = [np.concatenate([x, np.zeros((max_len - x.shape[0],) + x.shape[1:], dtype=dtype)], axis=0)
           for x in arr]
    if (min_length is not None) and ret[0].shape[0] < min_length:
        ret = [np.concatenate([x, np.zeros((min_length - x.shape[0],) + x.shape[1:], dtype=dtype)], axis=0)
               for x in ret]
    return np.array(ret), seq_length



############################################################################

# Fusion utils
def get_all_metadata(args):
    cxr_metadata = pd.read_csv(f'{args.cxr_data_root}/mimic-cxr-2.0.0-metadata.csv')
    icu_stay_metadata = pd.read_csv(f'{args.ehr_data_root}/root/all_stays.csv')
    columns = ['subject_id', 'stay_id', 'intime', 'outtime']
    
    # only common subjects with both icu stay and an xray
    # Note that inner merge includes rows if a chest X-ray is associated with multiple stays
    cxr_merged_icustays = cxr_metadata.merge(icu_stay_metadata[columns], how='inner', on='subject_id')
    # combine study date time
    cxr_merged_icustays['StudyTime'] = cxr_merged_icustays['StudyTime'].apply(lambda x: f'{int(float(x)):06}' )
    cxr_merged_icustays['StudyDateTime'] = pd.to_datetime(cxr_merged_icustays['StudyDate'].astype(str) + ' ' + cxr_merged_icustays['StudyTime'].astype(str) ,format="%Y%m%d %H%M%S")

    cxr_merged_icustays.intime=pd.to_datetime(cxr_merged_icustays.intime)
    cxr_merged_icustays.outtime=pd.to_datetime(cxr_merged_icustays.outtime)
    
    cxr_merged_icustays['time_diff'] = cxr_merged_icustays.StudyDateTime-cxr_merged_icustays.intime
    cxr_merged_icustays['time_diff'] = cxr_merged_icustays['time_diff'].apply(lambda x: np.round(x.total_seconds()/60/60,3))

    cxr_merged_icustays['full_stay_time'] = cxr_merged_icustays.outtime-cxr_merged_icustays.intime
    cxr_merged_icustays['full_stay_time'] = cxr_merged_icustays['full_stay_time'].apply(lambda x: np.round(x.total_seconds()/60/60,3))

    return cxr_merged_icustays 

def get_recent_cxr(cxr_merged_icustays_AP):
    groups = cxr_merged_icustays_AP.groupby('stay_id')
    groups_selected = []
    for group in groups:
        # select the latest cxr for the icu stay
        selected = group[1].sort_values('StudyDateTime').tail(1).reset_index()
        groups_selected.append(selected)
    groups = pd.concat(groups_selected, ignore_index=True)
    groups['lower'] = 0
    groups['upper'] = groups.full_stay_time
    
    return groups

def get_all_cxr(cxr_merged_icustays_AP):
    groups = cxr_merged_icustays_AP.groupby('study_id').first()
    groups = groups.reset_index()
    groups = groups.groupby('study_id').first().sort_values(by=['stay_id','StudyDateTime'])
    groups = groups.reset_index()    
    groups['lower'] = 0
    groups['upper'] = groups.time_diff
    
    return groups 

def load_decompensation_meta(args):
    
    cxr_merged_icustays = get_all_metadata(args)
    train_listfile = pd.read_csv(f'/scratch/se1525/mml-ssl/{args.task}/train_listfile.csv')
    val_listfile = pd.read_csv(f'/scratch/se1525/mml-ssl/{args.task}/val_listfile.csv')
    test_listfile = pd.read_csv(f'/scratch/se1525/mml-ssl/{args.task}/test_listfile.csv')
    
    # double check that this line does not cause issues
    listfile = train_listfile.append([val_listfile, test_listfile])
    listfile['subject_id'] = listfile['stay'].apply(lambda x: x.split("_")[0])
    listfile['subject_id'] = listfile['subject_id'].astype('int64')
    columns = ['subject_id', 'endtime']
    cxr_merged_icustays = cxr_merged_icustays.merge(listfile[columns], how='inner', on='subject_id')
    cxr_merged_icustays.endtime=pd.to_datetime(cxr_merged_icustays.endtime)
    cxr_merged_icustays_during = cxr_merged_icustays.loc[((cxr_merged_icustays.StudyDateTime>=cxr_merged_icustays.intime)&\
                                                          (cxr_merged_icustays.StudyDateTime<=cxr_merged_icustays.endtime))]
    
    cxr_merged_icustays_AP = cxr_merged_icustays_during[cxr_merged_icustays_during['ViewPosition'] == 'AP']
    groups = get_recent_cxr(cxr_merged_icustays_AP)

    return groups

    
def load_los_meta(args):

    cxr_merged_icustays = get_all_metadata(args)
    train_listfile = pd.read_csv(f'/scratch/se1525/mml-ssl/{args.task}/train_listfile.csv')
    val_listfile = pd.read_csv(f'/scratch/se1525/mml-ssl/{args.task}/val_listfile.csv')
    test_listfile = pd.read_csv(f'/scratch/se1525/mml-ssl/{args.task}/test_listfile.csv')
    
    # double check that this line does not cause issues
    listfile = train_listfile.append([val_listfile, test_listfile])
    listfile['subject_id'] = listfile['stay'].apply(lambda x: x.split("_")[0])
    listfile['subject_id'] = listfile['subject_id'].astype('int64')
    columns = ['subject_id', 'endtime']
    cxr_merged_icustays = cxr_merged_icustays.merge(listfile[columns], how='inner', on='subject_id')
    cxr_merged_icustays.endtime=pd.to_datetime(cxr_merged_icustays.endtime)

    cxr_merged_icustays_during = cxr_merged_icustays.loc[((cxr_merged_icustays.StudyDateTime>=cxr_merged_icustays.intime)&\
                                                            (cxr_merged_icustays.StudyDateTime<=cxr_merged_icustays.endtime))]

    cxr_merged_icustays_AP = cxr_merged_icustays_during[cxr_merged_icustays_during['ViewPosition'] == 'AP']
    groups = get_recent_cxr(cxr_merged_icustays_AP)

    return groups

def load_mortality_meta(args):
    cxr_merged_icustays = get_all_metadata(args)
    end_time = cxr_merged_icustays.intime + pd.DateOffset(hours=48)
    cxr_merged_icustays_during = cxr_merged_icustays.loc[(cxr_merged_icustays.StudyDateTime>=cxr_merged_icustays.intime)&\
                                                         ((cxr_merged_icustays.StudyDateTime<=end_time))]
    cxr_merged_icustays_AP = cxr_merged_icustays_during[cxr_merged_icustays_during['ViewPosition'] == 'AP']
    groups = get_recent_cxr(cxr_merged_icustays_AP)

    return groups


def load_phenotyping_meta(args):
    cxr_merged_icustays = get_all_metadata(args)
    end_time = cxr_merged_icustays.outtime
    cxr_merged_icustays_during = cxr_merged_icustays.loc[(cxr_merged_icustays.StudyDateTime>=cxr_merged_icustays.intime)&\
                                                         ((cxr_merged_icustays.StudyDateTime<=end_time))]
    cxr_merged_icustays_AP = cxr_merged_icustays_during[cxr_merged_icustays_during['ViewPosition'] == 'AP']
    groups = get_recent_cxr(cxr_merged_icustays_AP)

    return groups

def load_readmission_meta(args):
    cxr_merged_icustays = get_all_metadata(args)
    end_time = cxr_merged_icustays.outtime
    cxr_merged_icustays_during = cxr_merged_icustays.loc[(cxr_merged_icustays.StudyDateTime>=cxr_merged_icustays.intime)&\
                                                         ((cxr_merged_icustays.StudyDateTime<=end_time))]
    cxr_merged_icustays_AP = cxr_merged_icustays_during[cxr_merged_icustays_during['ViewPosition'] == 'AP']
    groups = get_recent_cxr(cxr_merged_icustays_AP)

    return groups

def load_radiology_meta(args):
    cxr_merged_icustays = get_all_metadata(args)
    end_time = cxr_merged_icustays.outtime
    cxr_merged_icustays_during = cxr_merged_icustays.loc[(cxr_merged_icustays.StudyDateTime>=cxr_merged_icustays.intime)&\
                                                        ((cxr_merged_icustays.StudyDateTime<=end_time))]
    cxr_merged_icustays_AP = cxr_merged_icustays_during[cxr_merged_icustays_during['ViewPosition'] == 'AP']
    groups = get_all_cxr(cxr_merged_icustays_AP)

    return groups

#NOTE: only if pretraining with large dataset, i.e. logic should be added to have a 'pretraining' task if args.dataset='all'
def load_pretraining_meta(args):
    cxr_merged_icustays = get_all_metadata(args)
    cxr_merged_icustays_AP = cxr_merged_icustays[cxr_merged_icustays['ViewPosition'] == 'AP']
    groups = cxr_merged_icustays_AP
    return groups

# check the task names to match the dictionary 
# check pre-training settings per task for supervised 
load_task_meta = {'decompensation':load_decompensation_meta,
                  'length-of-stay':load_los_meta,
                  'phenotyping':load_phenotyping_meta,
                  'in-hospital-mortality':load_mortality_meta,
                  'readmission':load_readmission_meta,
                  'radiology':load_radiology_meta,
                  'pretraining':load_pretraining_meta}


def count_labels(dataset):
    total_labels_count = 0
    for key, value in dataset.ehr_ds.data_map.items():
        # Access the labels for each key
        labels = value['labels']
        # Now you can work with the labels, for example, counting positive labels
        for label in labels:
            if label == 1:  # Assuming 1 indicates a positive label
                total_labels_count += 1 
    return total_labels_count




def get_final_meta(args):

        # Load cxr and ehr groups
    cxr_merged_icustays = load_task_meta[args.task](args) 

    # Add the labels 
    splits_labels_train = pd.read_csv(f'{args.ehr_data_root}/{args.task}/train_listfile.csv')
    splits_labels_val = pd.read_csv(f'{args.ehr_data_root}/{args.task}/val_listfile.csv')
    splits_labels_test = pd.read_csv(f'{args.ehr_data_root}/{args.task}/test_listfile.csv')

    #TODO: investigate why total size of cxr_merged_icustays drops after the three steps below
    train_meta_with_labels = cxr_merged_icustays.merge(splits_labels_train, how='inner', on='stay_id')#change dataset size here
    val_meta_with_labels = cxr_merged_icustays.merge(splits_labels_val, how='inner', on='stay_id')
    test_meta_with_labels = cxr_merged_icustays.merge(splits_labels_test, how='inner', on='stay_id')

    # Get rid of chest X-rays that don't have radiology reports
    metadata = pd.read_csv(f'{args.cxr_data_root}/mimic-cxr-2.0.0-metadata.csv')
    labels = pd.read_csv(f'{args.cxr_data_root}/mimic-cxr-2.0.0-chexpert.csv')
    metadata_with_labels = metadata.merge(labels[['study_id']], how='inner', on='study_id').drop_duplicates(subset=['dicom_id'])
    train_meta_with_labels = train_meta_with_labels.merge(metadata_with_labels[['dicom_id']], how='inner', on='dicom_id')
    val_meta_with_labels = val_meta_with_labels.merge(metadata_with_labels[['dicom_id']], how='inner', on='dicom_id')
    test_meta_with_labels = test_meta_with_labels.merge(metadata_with_labels[['dicom_id']], how='inner', on='dicom_id')

    return train_meta_with_labels, val_meta_with_labels, test_meta_with_labels



def load_cxr_ehr(args, 
                 ehr_train_ds, 
                 ehr_val_ds, 
                 cxr_train_ds, 
                 cxr_val_ds, 
                 ehr_test_ds, 
                 cxr_test_ds):

    train_meta_with_labels, val_meta_with_labels, test_meta_with_labels = get_final_meta(args) 

    # Multimodal class
    train_ds = MIMIC_CXR_EHR(args, train_meta_with_labels, ehr_train_ds, cxr_train_ds)
    # total_labels_count = count_labels(train_ds) # this does nothing

    val_ds = MIMIC_CXR_EHR(args, val_meta_with_labels, ehr_val_ds, cxr_val_ds, split='val')
    # total_labels_count = count_labels(val_ds) # this does nothing

    test_ds = MIMIC_CXR_EHR(args, test_meta_with_labels, ehr_test_ds, cxr_test_ds, split='test')
    # total_labels_count = count_labels(test_ds) # this does nothing

    collate = my_collate_fusion
    train_dl = DataLoader(train_ds, args.batch_size, shuffle=True, collate_fn=collate, drop_last=True) 
    val_dl = DataLoader(val_ds, args.batch_size, shuffle=False, collate_fn=collate, drop_last=False) 
    test_dl = DataLoader(test_ds, args.batch_size, shuffle=False, collate_fn=collate, drop_last=False)
    return train_dl, val_dl, test_dl





def my_collate_fusion(batch):
    x = [item[0] for item in batch]
    pairs = [False if item[1] is None else True for item in batch]
    img = torch.stack([torch.zeros(3, 224, 224) if item[1] is None else item[1] for item in batch])
    x, seq_length = pad_zeros(x)
    targets_ehr = np.array([item[2] for item in batch])
    targets_cxr = torch.stack([torch.zeros(14) if item[3] is None else item[3] for item in batch])
    return [x, img, targets_ehr, targets_cxr, seq_length, pairs]
    
