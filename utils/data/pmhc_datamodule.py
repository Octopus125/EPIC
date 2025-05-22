import pandas as pd
import json
import random
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

class PMHCPredDataset(Dataset):
    def __init__(self, binding_file='./data/pMHC_binding.csv', mhc_seq_file='./data/MHC_seq_dict.json'):
        with open(mhc_seq_file, 'r') as f:
            self.mhc_seq_dict = json.load(f)
        
        all_annotations = pd.read_csv(binding_file)
        positive_samples = all_annotations[all_annotations['label'] == 1]
        negative_samples = all_annotations[all_annotations['label'] == 0]

        self.positive_mhc_name_list = positive_samples['MHC_name'].tolist()
        self.positive_pep_list = positive_samples['pep'].tolist()
        self.negative_mhc_name_list = negative_samples['MHC_name'].tolist()
        self.negative_pep_list = negative_samples['pep'].tolist()        


    def __getitem__(self, index): 
        if index % 2 == 0:
            index = index // 2
            pep = self.positive_pep_list[index]
            mhc_name = self.positive_mhc_name_list[index]
            mhc_seq = self.mhc_seq_dict[mhc_name]
            label = 1
        else:
            negative_index = random.randint(0, len(self.negative_mhc_name_list)-1)
            pep = self.negative_pep_list[negative_index]
            mhc_name = self.negative_mhc_name_list[negative_index]
            mhc_seq = self.mhc_seq_dict[mhc_name]
            label = 0

        return {'mhc_seq': mhc_seq, 'pep_seq': pep, 'label': label}

    def __len__(self):
        length = len(self.positive_mhc_name_list) * 2
        return length

class PMHCPredDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 256):
        super().__init__()
        self.batch_size = batch_size
        train_binding_file = './data/split/pMHC_binding_train.csv'
        valid_binding_file = './data/split/pMHC_binding_valid.csv'
        mhc_seq_file ='./data/MHC_seq_dict.json'

        self.train_dataset = PMHCPredDataset(binding_file=train_binding_file, mhc_seq_file=mhc_seq_file)
        self.valid_dataset = PMHCPredDataset(binding_file=valid_binding_file, mhc_seq_file=mhc_seq_file)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size)