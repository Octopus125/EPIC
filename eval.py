from Bio import pairwise2
from Bio.Align import substitution_matrices
import pandas as pd
import os
from tqdm import tqdm

# Define the alignment matrix
matrix = substitution_matrices.load("BLOSUM62")

def calculate_uniqueness(peptides):
    unique_peptides = set(peptides)
    return len(unique_peptides) / len(peptides) if peptides else 0

def calculate_similarity(seq1, seq2):
    alignments = pairwise2.align.globalxx(seq1, seq2)
    if alignments:
        best_alignment = alignments[0]
        score = best_alignment.score
        max_score = max(len(seq1), len(seq2))  
        similarity = score / max_score
        return similarity
    return 0

def calculate_diversity(peptides):
    if len(peptides) < 2:
        return 0
    total_pairs = 0
    similarity_sum = 0
    for i in range(len(peptides)):
        for j in range(i + 1, len(peptides)):
            similarity = calculate_similarity(peptides[i], peptides[j])
            total_pairs += 1
            similarity_sum += similarity
    return 1 - similarity_sum / total_pairs if total_pairs else 0

def calculate_novelty(generated_peptides, reference_peptides):
    if not generated_peptides or not reference_peptides:
        return 0
    total_comparisons = 0
    similarity_sum = 0
    for gen_peptide in generated_peptides:
        for ref_peptide in reference_peptides:
            similarity = calculate_similarity(gen_peptide, ref_peptide)
            total_comparisons += 1
            similarity_sum += similarity
    return similarity_sum / total_comparisons if total_comparisons else 0


def main(result_dir):
    reference_data = pd.read_csv('./data/test.csv')
    all_results = []

    for tcr_id in tqdm(reference_data['TCR_id'].unique()):
        file_name = f"{tcr_id}.csv"
        file_path = os.path.join(result_dir, file_name)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            generated_peptides = df['generated_peptides'].tolist()
            reference_peptides = reference_data[reference_data['TCR_id'] == tcr_id]['pep'].tolist()

            uniqueness = calculate_uniqueness(generated_peptides)
            novelty = calculate_novelty(generated_peptides, reference_peptides)
            diversity = calculate_diversity(generated_peptides)

            result_dict = {
                'TCR_id': tcr_id,
                'Uniqueness': uniqueness,
                'Novelty': novelty,
                'Diversity': diversity
            }
            all_results.append(result_dict)
        else:
            print(f"File {file_path} is not exist.")

    final_result = pd.DataFrame(all_results)

    avg_row = final_result.drop(columns='TCR_id').mean()
    avg_row['TCR_id'] = 'Average'
    final_result = pd.concat([final_result, avg_row.to_frame().T], ignore_index=True)

    csv_file_path = os.path.join(result_dir, 'result.csv')
    final_result.to_csv(csv_file_path, index=False)
    print(final_result)

if __name__ == "__main__":
    # result_dir = './results/cond_resample_scale_50.0'
    result_dir = './results/random'
    main(result_dir)