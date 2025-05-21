import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split

def split_pmhc_data():
    pmhc_data_all_file = './data/pMHC_binding.csv'
    train_file = './data/split/pMHC_binding_train.csv'
    valid_file = './data/split/pMHC_binding_valid.csv'

    all_annotations = pd.read_csv(pmhc_data_all_file)
    mhc_name_list = all_annotations['MHC_name'].tolist()
    pep_list = all_annotations['pep'].tolist()
    label_list = all_annotations['label'].tolist()

    mhc_groups = defaultdict(list)
    for i, mhc in enumerate(mhc_name_list):
        mhc_groups[mhc].append(i)

    train_indices = []
    valid_indices = []
    for mhc, indices in mhc_groups.items():
        labels = [label_list[i] for i in indices]
        if len(indices) > 10:
            # print(labels)
            train_idx, valid_idx, train_labels, valid_labels = train_test_split(indices, labels, test_size=0.1, stratify=labels, random_state=42)
        else:
            train_idx = indices[1:]
            valid_idx = indices[0:1]
        train_indices.extend(train_idx)
        valid_indices.extend(valid_idx)
        print(f'{mhc}: {len(indices)} in total, {len(train_idx)} train, {len(valid_idx)} valid')

    train_mhc_name_list = [mhc_name_list[i] for i in train_indices]
    train_pep_list = [pep_list[i] for i in train_indices]
    train_label_list = [label_list[i] for i in train_indices]

    valid_mhc_name_list = [mhc_name_list[i] for i in valid_indices]
    valid_pep_list = [pep_list[i] for i in valid_indices]
    valid_label_list = [label_list[i] for i in valid_indices]

    train_data = {'MHC_name': train_mhc_name_list, 'pep': train_pep_list, 'label': train_label_list}
    valid_data = {'MHC_name': valid_mhc_name_list, 'pep': valid_pep_list, 'label': valid_label_list}
    pd.DataFrame(train_data).to_csv(train_file, index=False)
    pd.DataFrame(valid_data).to_csv(valid_file, index=False)

    print(f'train data total: {len(train_mhc_name_list)}')
    print(f'valid data total: {len(valid_mhc_name_list)}')

def split_tcr_speci_data():
    data_all_file = './data/TCR_pep_specificity.csv'
    train_file = './data/split/TCR_pep_specificity_train.csv'
    valid_file = './data/split/TCR_pep_specificity_valid.csv'

    all_annotations = pd.read_csv(data_all_file)
    tcr_name_list = all_annotations['TCR_id'].tolist()
    pep_list = all_annotations['peptide'].tolist()
    specificity_score_list = all_annotations['specificity_score'].tolist()
    specificity_label_list = [1 if score >= 50 else 0 for score in specificity_score_list]

    tcr_alpha_seq_list = all_annotations['TCR_alpha_seq'].tolist()
    tcr_beta_seq_list = all_annotations['TCR_beta_seq'].tolist()

    pep_groups = defaultdict(list)
    for i, pep in enumerate(pep_list):
        pep_groups[pep].append(i)

    train_indices = []
    valid_indices = []
    for pep, indices in pep_groups.items():
        labels = [specificity_label_list[i] for i in indices]
        if len(indices) > 10:
            train_idx, valid_idx, train_labels, valid_labels = train_test_split(indices, labels, test_size=0.1, stratify=labels, random_state=42)
        else:
            train_idx = indices[1:]
            valid_idx = indices[0:1]
        train_indices.extend(train_idx)
        valid_indices.extend(valid_idx)
        print(f'{pep}: {len(indices)} in total, {len(train_idx)} train, {len(valid_idx)} valid')

    train_tcr_name_list = [tcr_name_list[i] for i in train_indices]
    train_pep_list = [pep_list[i] for i in train_indices]
    train_specificity_score_list = [specificity_score_list[i] for i in train_indices]
    train_tcr_alpha_seq_list = [tcr_alpha_seq_list[i] for i in train_indices]
    train_tcr_beta_seq_list = [tcr_beta_seq_list[i] for i in train_indices]

    valid_tcr_name_list = [tcr_name_list[i] for i in valid_indices]
    valid_pep_list = [pep_list[i] for i in valid_indices]
    valid_specificity_score_list = [specificity_score_list[i] for i in valid_indices]
    valid_tcr_alpha_seq_list = [tcr_alpha_seq_list[i] for i in valid_indices]
    valid_tcr_beta_seq_list = [tcr_beta_seq_list[i] for i in valid_indices]
    

    train_data = {'TCR_id': train_tcr_name_list, 'peptide': train_pep_list, 'specificity_score': train_specificity_score_list,
                  'TCR_alpha_seq': train_tcr_alpha_seq_list, 'TCR_beta_seq': train_tcr_beta_seq_list}
    valid_data = {'TCR_id': valid_tcr_name_list, 'peptide': valid_pep_list, 'specificity_score': valid_specificity_score_list,
                  'TCR_alpha_seq': valid_tcr_alpha_seq_list, 'TCR_beta_seq': valid_tcr_beta_seq_list}

    pd.DataFrame(train_data).to_csv(train_file, index=False)
    pd.DataFrame(valid_data).to_csv(valid_file, index=False)

    print(f'train data total: {len(train_tcr_name_list)}')
    print(f'valid data total: {len(valid_tcr_name_list)}')
