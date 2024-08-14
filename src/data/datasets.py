
import os
import torch
import random
import numpy as np
import pandas as pd 

from PIL import Image
from torch.utils.data import Dataset

R_CLASSES  = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
       'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
       'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other',
       'Pneumonia', 'Pneumothorax', 'Support Devices']

CLASSES = [
       'Acute and unspecified renal failure', 'Acute cerebrovascular disease',
       'Acute myocardial infarction', 'Cardiac dysrhythmias',
       'Chronic kidney disease',
       'Chronic obstructive pulmonary disease and bronchiectasis',
       'Complications of surgical procedures or medical care',
       'Conduction disorders', 'Congestive heart failure; nonhypertensive',
       'Coronary atherosclerosis and other heart disease',
       'Diabetes mellitus with complications',
       'Diabetes mellitus without complication',
       'Disorders of lipid metabolism', 'Essential hypertension',
       'Fluid and electrolyte disorders', 'Gastrointestinal hemorrhage',
       'Hypertension with complications and secondary hypertension',
       'Other liver diseases', 'Other lower respiratory disease',
       'Other upper respiratory disease',
       'Pleurisy; pneumothorax; pulmonary collapse',
       'Pneumonia (except that caused by tuberculosis or sexually transmitted disease)',
       'Respiratory failure; insufficiency; arrest (adult)',
       'Septicemia (except in labor)', 
       'Shock']

# CXR datset
class MIMICCXR(Dataset):
    def __init__(self, paths, args, transform=None, split='train'):
        self.data_dir = args.cxr_data_root
        self.args = args
        self.CLASSES  = R_CLASSES
        self.filenames_to_path = {path.split('/')[-1].split('.')[0]: path for path in paths}

        metadata = pd.read_csv(f'{self.data_dir}/mimic-cxr-2.0.0-metadata.csv')
        labels = pd.read_csv(f'{self.data_dir}/mimic-cxr-2.0.0-chexpert.csv')
        labels[self.CLASSES] = labels[self.CLASSES].fillna(0)
        labels = labels.replace(-1.0, 0.0)
        
        splits = pd.read_csv(f'{self.data_dir}/mimic-cxr-ehr-split.csv')


        metadata_with_labels = metadata.merge(labels[self.CLASSES+['study_id'] ], how='inner', on='study_id')


        self.filesnames_to_labels = dict(zip(metadata_with_labels['dicom_id'].values, metadata_with_labels[self.CLASSES].values))
        self.filenames_loaded = splits.loc[splits.split==split]['dicom_id'].values
        self.transform = transform
        self.filenames_loaded = [filename  for filename in self.filenames_loaded if filename in self.filesnames_to_labels]

    def __getitem__(self, index):
        # why was this commented out? 
        if isinstance(index, str):
            img = Image.open(self.filenames_to_path[index]).convert('RGB')
            labels = torch.tensor(self.filesnames_to_labels[index]).float()
            if self.transform is not None:
                img = self.transform(img)
            return img, labels
          
        
        filename = self.filenames_loaded[index]
        img = Image.open(self.filenames_to_path[filename]).convert('RGB')
        labels = torch.tensor(self.filesnames_to_labels[filename]).float()

        if self.transform is not None:
            img = self.transform(img)
        return img, labels

    def __len__(self):
        return len(self.filenames_loaded)


############################################################################
# EHR dataset
class EHRdataset(Dataset):
    def __init__(self, args, discretizer, normalizer, listfile, dataset_dir, return_names=True, period_length=48.0, transforms=None):
        self.return_names = return_names
        self.discretizer = discretizer
        self.normalizer = normalizer
        self._period_length = period_length
        self.args=args

        self._dataset_dir = dataset_dir
        listfile_path = listfile
        with open(listfile_path, "r") as lfile:
            self._data = lfile.readlines()
        self._listfile_header = self._data[0]
        self.CLASSES = self._listfile_header.strip().split(',')[3:]
        self._data = self._data[1:]
        self.transforms = transforms

        self._data = [line.split(',') for line in self._data]
        if self.args.task=='length-of-stay' or self.args.task=='decompensation':
            self.data_map = {
                (mas[0],float(mas[1])): {
                    'labels': list(map(float, mas[3:])),
                    'stay_id': float(mas[2]),
                    'time': float(mas[1]),
                    }
                for mas in self._data
                    
                }
        else:
            self.data_map = {
                mas[0]: {
                    'labels': list(map(float, mas[3:])),
                    'stay_id': float(mas[2]),
                    'time': float(mas[1]),
                    }
                for mas in self._data
            }

        self.names = list(self.data_map.keys())
        self.times= None
    
    def read_chunk(self, chunk_size):
        data = {}
        for i in range(chunk_size):
            ret = reader.read_next()
            for k, v in ret.items():
                if k not in data:
                    data[k] = []
                data[k].append(v)
        data["header"] = data["header"][0]
        return data

    def _read_timeseries(self, ts_filename, lower_bound, upper_bound):
        
        ret = []
        with open(os.path.join(self._dataset_dir, ts_filename), "r") as tsfile:
            header = tsfile.readline().strip().split(',')
            assert header[0] == "Hours"
            for line in tsfile:
                mas = line.strip().split(',')
                t = float(mas[0])
                if t < lower_bound:
                    continue
                elif (t> lower_bound) & (t <upper_bound) :
                    ret.append(np.array(mas))
                elif t > upper_bound:
                    break
        try:
            # print("Hour", upper_bound)
            # print("EHR data", np.stack(ret))
            return (np.stack(ret), header)
        except ValueError:
            print("exception in read_timeseries")
            ret = ([['0.11666666666666667', '', '', '', '', '', '', '', '', '109', '',
                     '', '', '30', '', '', '', ''],
                    ['0.16666666666666666', '', '61.0', '', '', '', '', '', '', '109',
                    '', '64', '97.0', '29', '74.0', '', '', '']])
            # print(ts_filename, lower_bound, upper_bound)
            return (np.stack(ret), header)
    
    def read_by_file_name(self, index, time, lower_bound, upper_bound):
        if self.args.task=='length-of-stay' or self.args.task=='decompensation':
            t = self.data_map[(index,time)]['time'] 
            y = self.data_map[(index,time)]['labels']
            stay_id = self.data_map[(index,time)]['stay_id']
            (X, header) = self._read_timeseries(index, lower_bound=lower_bound, upper_bound=time)
        else:
            t = self.data_map[index]['time'] 
            y = self.data_map[index]['labels']
            stay_id = self.data_map[index]['stay_id']
            (X, header) = self._read_timeseries(index, lower_bound=lower_bound, upper_bound=upper_bound)

        return {"X": X,
                "t": t,
                "y": y,
                'stay_id': stay_id,
                "header": header,
                "name": index}

    def __getitem__(self, item_args, lower, upper):
        if self.args.task=='length-of-stay' or self.args.task=='decompensation':
            time = item_args[1]
            index = item_args[0]
        else:
            index = item_args
            if isinstance(index, int):
                index = self.names[index]
            time = None
        ret = self.read_by_file_name(index, time, lower, upper)
        data = ret["X"]
        # ts = data.shape[0]
        ts = ret["t"] if ret['t'] > 0.0 else self._period_length    
        ys = ret["y"]
        names = ret["name"]


        data = self.discretizer.transform(data, end=ts)[0]
        if (self.normalizer is not None):
            data = self.normalizer.transform(data)

        
        if 'length-of-stay' in self._dataset_dir:
            ys = np.array(ys, dtype=np.float32) if len(ys) > 1 else np.array(ys, dtype=np.float32)[0]
        else:
            ys = np.array(ys, dtype=np.int32) if len(ys) > 1 else np.array(ys, dtype=np.int32)[0]
        return data, ys

    
    def __len__(self):
        return len(self.names)
    


############################################################################
# Fusion dataset

class MIMIC_CXR_EHR(Dataset):
    def __init__(self, args, metadata_with_labels, ehr_ds, cxr_ds, split='train'):
        
        if 'radiology' in args.labels_set:
            self.CLASSES = R_CLASSES
        else:
            self.CLASSES = CLASSES
        
        self.metadata_with_labels = metadata_with_labels
        self.cxr_files_paired = self.metadata_with_labels.dicom_id.values
        self.ehr_files_paired = (self.metadata_with_labels['stay'].values)
        self.time_diff = self.metadata_with_labels.time_diff
        self.lower = self.metadata_with_labels.lower
        self.upper = self.metadata_with_labels.upper
        self.cxr_files_all = cxr_ds.filenames_loaded
        self.ehr_files_all = ehr_ds.names
        
        self.ehr_files_unpaired = list(set(self.ehr_files_all) - set(self.ehr_files_paired))
        
        self.ehr_ds = ehr_ds
        self.cxr_ds = cxr_ds

        if args.task == 'decompensation' or args.task == 'length-of-stay':
            self.paired_times= (self.metadata_with_labels['period_length'].values)
            self.ehr_paired_list = list(zip(self.ehr_files_paired, self.paired_times))
        
        self.args = args
        self.split = split  
        self.data_ratio = self.args.data_ratio if split=='train' else 1.0

        self.get_data = {'paired':self._get_paired,
                         'ehr_only':self._get_ehr_only,
                         'radiology':self._get_radiology,
                         'joint_ehr':self._get_joint_ehr
                        }
        
        self.get_len = {'paired':len(self.ehr_files_paired),
                         'ehr_only':len(self.ehr_files_all),
                         'radiology':len(self.cxr_files_all),
                         'joint_ehr':len(self.ehr_files_paired) + int(self.data_ratio * len(self.ehr_files_unpaired))
                        }
    def __getitem__(self, index):
        ehr_data, cxr_data, labels_ehr, labels_cxr = self.get_data[self.args.data_pairs](index)
        return ehr_data, cxr_data, labels_ehr, labels_cxr


    def _get_joint_ehr(self,index):
        if index < len(self.ehr_files_paired):
            ehr_data, labels_ehr = self.ehr_ds[self.ehr_files_paired[index]]
            cxr_data, labels_cxr = self.cxr_ds[self.cxr_files_paired[index]]
        else:
            index = random.randint(0, len(self.ehr_files_unpaired)-1) 
            if self.args.task == 'decompensation' or self.args.task == 'length-of-stay':
                ehr_data, labels_ehr = self.ehr_ds[self.ehr_files_paired[index]]
            else:
                ehr_data, labels_ehr = self.ehr_ds[self.ehr_files_paired[index]]
            cxr_data, labels_cxr = None, None
        return ehr_data, cxr_data, labels_ehr, labels_cxr

    def _get_paired(self,index):
        cxr_data, labels_cxr = self.cxr_ds[self.cxr_files_paired[index]]
        lower = self.metadata_with_labels.iloc[index].lower
        upper = self.metadata_with_labels.iloc[index].upper
        if self.args.task == 'decompensation' or self.args.task == 'length-of-stay':
            ehr_data, labels_ehr = self.ehr_ds.__getitem__(self.ehr_paired_list[index], lower, upper)
        else:
            ehr_data, labels_ehr = self.ehr_ds.__getitem__(self.ehr_files_paired[index],lower,upper)
        return ehr_data, cxr_data, labels_ehr, labels_cxr
    
    # changed to return 0 EHR data
    def _get_radiology(self,index):
        ehr_data, labels_ehr = np.zeros((1, 10)), np.zeros(self.args.num_classes)
        cxr_data, labels_cxr = self.cxr_ds[self.cxr_files_all[index]]
        return ehr_data, cxr_data, labels_ehr, labels_cxr
    
    def _get_ehr_only(self,index):
        ehr_data, labels_ehr = self.ehr_ds[self.ehr_files_all[index]]
        cxr_data, labels_cxr = None, None
        return ehr_data, cxr_data, labels_ehr, labels_cxr

  
    def __len__(self):
            return self.get_len[self.args.data_pairs]











