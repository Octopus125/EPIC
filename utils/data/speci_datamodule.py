import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

class SpeciPredDataset(Dataset):
    def __init__(self, seq_file='./data/TCR_pep_specificity.csv'):
        all_annotations = pd.read_csv(seq_file)
        self.tcr_id_list = all_annotations['TCR_id'].tolist()
        self.pep_list = all_annotations['peptide'].tolist()
        self.specificity_score_list = all_annotations['specificity_score'].tolist()
        self.specificity_label_list = [1 if score >= 50 else 0 for score in self.specificity_score_list]

        # sequence
        tcr_alpha_seq_list = all_annotations['TCR_alpha_seq'].tolist()
        tcr_beta_seq_list = all_annotations['TCR_beta_seq'].tolist()
        self.tcr_seq_list = [tra + '.' + trb for tra, trb in zip(tcr_alpha_seq_list, tcr_beta_seq_list)]

    def __getitem__(self, index):
        pep = self.pep_list[index]
        tcr_seq = self.tcr_seq_list[index]
        label = self.specificity_label_list[index]
        return {'tcr_seq': tcr_seq, 'pep_seq': pep, 'label': label}

    def __len__(self):
        return len(self.tcr_id_list)

class SpeciPredDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 256):
        super().__init__()
        self.batch_size = batch_size
        train_file = './data/split/TCR_pep_specificity_train.csv'
        valid_file = './data/split/TCR_pep_specificity_valid.csv'

        self.train_dataset = SpeciPredDataset(seq_file=train_file)
        self.valid_dataset = SpeciPredDataset(seq_file=valid_file)
    
    def train_dataloader(self):
        weights = []
        for i in range(len(self.train_dataset)):
            if self.train_dataset.specificity_label_list[i] == 1:
                weights.append(0.9)
            else:
                weights.append(0.1)
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights))
        return DataLoader(self.train_dataset, batch_size=self.batch_size, sampler=sampler)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size)