import numpy as np
import pandas as pd
from collections import Counter

esm_dict = {'<cls>': 0, '<pad>': 1, '<eos>': 2, '<unk>': 3, 
 'L': 4, 'A': 5, 'G': 6, 'V': 7, 'S': 8, 'E': 9, 'R': 10, 'T': 11, 'I': 12, 'D': 13, 'P': 14, 'K': 15, 'Q': 16, 'N': 17, 
'F': 18, 'Y': 19, 'M': 20, 'H': 21, 'W': 22, 'C': 23, 'X': 24, 'B': 25, 'U': 26, 'Z': 27, 'O': 28,
 '.': 29, '-': 30, '<null_1>': 31, '<mask>': 32}

def get_peptide_statis(max_seq_length=15):
    pep_file = './data/peptides.csv'
    data_df = pd.read_csv(pep_file)
    pep_list = data_df['peptide'].tolist()
    label_list = data_df['label'].tolist()

    agpep_list = [pep for pep, label in zip(pep_list, label_list) if label == 1]
    seq_length = np.zeros(max_seq_length + 1)
    aa_count = np.zeros(len(esm_dict))

    for agpep in agpep_list:
        length = len(agpep)
        seq_length[length] += 1

        for aa in agpep:
            aa_count[esm_dict[aa]] += 1
    
    seq_length_freq = seq_length / sum(seq_length)
    aa_count_freq = aa_count / sum(aa_count)
    
    return seq_length_freq, aa_count_freq


seq_length_freq, aa_count_freq = get_peptide_statis()

def get_mhc_statis():
    pMHC_file = './data/pMHC_binding.csv'
    data_df = pd.read_csv(pMHC_file)
    mhc_list = data_df['MHC_name'].tolist()
    label_list = data_df['label'].tolist()

    pos_mhc_list = [mhc for mhc, label in zip(mhc_list, label_list) if label == 1]
    mhc_counts = Counter(pos_mhc_list)
    return mhc_counts

mhc_counts = get_mhc_statis()