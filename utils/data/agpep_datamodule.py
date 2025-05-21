import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import Dataset, Subset, DataLoader
from sklearn.model_selection import train_test_split

class AgpepPredDataset(Dataset):
    def __init__(self, data_file='./data/peptides.csv'):
        data_df = pd.read_csv(data_file)
        self.pep_list = data_df['peptide'].tolist()
        self.label_list = data_df['label'].tolist()

    def __getitem__(self, index):
        pep = self.pep_list[index]
        label = self.label_list[index]
        return {'pep': pep, 'label': label}

    def __len__(self):
        return len(self.pep_list)

class AgpepPredDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 256):
        super().__init__()
        self.batch_size = batch_size
        data_file='./data/peptides.csv'

        self.dataset = AgpepPredDataset(data_file=data_file)
        self.split_dataset_stratified()

    def split_dataset_stratified(self):
        labels = self.dataset.label_list

        train_indices, valid_indices = train_test_split(
            range(len(self.dataset)), 
            test_size=0.1, 
            stratify=labels,
            random_state=42
        )

        self.train_dataset = Subset(self.dataset, train_indices)
        self.valid_dataset = Subset(self.dataset, valid_indices)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size)