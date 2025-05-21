import torch
import pandas as pd

class PeptideDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, max_len=14, vocab_size=32):
        self.data = pd.read_csv(csv_file)
        self.sequences = self.data['peptide'].values
        self.labels = self.data['label'].values
        self.sequences = [pep for pep, label in zip(self.sequences, self.labels) if label == 1]
        self.max_len = max_len

        self.esm_dict = {'<cls>': 0, '<pad>': 1, '<eos>': 2, '<unk>': 3, 
                         'L': 4, 'A': 5, 'G': 6, 'V': 7, 'S': 8, 'E': 9, 'R': 10, 'T': 11, 'I': 12, 'D': 13,
                         'P': 14, 'K': 15, 'Q': 16, 'N': 17, 'F': 18, 'Y': 19, 'M': 20, 'H': 21, 'W': 22, 'C': 23}
        self.vocab_size = len(self.esm_dict)
        
    def __len__(self):
        return len(self.sequences)
        
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        
        # transfer the sequence to index, add [CLS] token (index 0) at the beginning and [EOS] token (index 2) at the end
        seq_idx = [0] + [self.esm_dict.get(aa, 0) for aa in sequence] + [2]
        
        # Padding
        if len(seq_idx) < self.max_len + 2:  # +2 for [CLS] and [EOS]
            # add [PAD] token (index 1)
            pad_len = self.max_len + 2 - len(seq_idx)
            seq_idx = seq_idx + [1] * pad_len
            padding_mask = [0] * (len(sequence) + 2) + [1] * pad_len
        else:
            seq_idx = seq_idx[:self.max_len + 2]
            padding_mask = [0] * (self.max_len + 2)

        seq_tensor = torch.tensor(seq_idx, dtype=torch.long)
        padding_mask = torch.tensor(padding_mask, dtype=torch.bool)
        
        return seq_tensor, padding_mask