import pandas as pd
from tqdm import tqdm
import pickle
import os
import argparse
import torch
import torch.nn.functional as F

from utils.constant import *
from model.generator import DiffusionModel
from model.classifier.cond_classifier import ConditioningClassifier

def parse_args():
    modes = ['random', 'uncond', 'cond', 'cond_resample']
    parser = argparse.ArgumentParser(description='xxx')
    parser.add_argument('--mode', type=str, choices=modes, required=True, help='sample mode')
    parser.add_argument('--scale', type=float, default=50.0, help='guidance scale')
    parser.add_argument('--specifi', action='store_true', help='using specificity classifier')
    parser.add_argument('--agpep', action='store_true', help='using agpep classifier')
    parser.add_argument('--pmhc', action='store_true', help='using pmhc binding classifier')
    
    args = parser.parse_args()
    flag = any([args.specifi, args.agpep, args.pmhc])
    args.specifi = args.specifi if flag else True
    args.agpep = args.agpep if flag else True
    args.pmhc = args.pmhc if flag else True

    remark = f'{args.mode}'
    if args.mode == 'cond' or args.mode == 'cond_resample':
        if flag:
            if args.specifi:
                remark += '_speci'
            if args.agpep:
                remark += '_agpep'
            if args.pmhc:
                remark += '_pmhc'
        remark += '_scale_' + str(args.scale)

    result_dir = './results/' + remark
    print(f'result_dir: {result_dir}')
    os.makedirs(result_dir, exist_ok=True)
    return args, result_dir


def logit_to_fasta(logit, num_samples):
    samples = []
    for i in range(num_samples):
        generated_indices = torch.argmax(logit[i], dim=-1).squeeze().cpu().numpy()
        idx_to_aa =  {v: k for k, v in esm_dict.items()}
        generated_sequence = ''.join(idx_to_aa[idx] for idx in generated_indices if idx >= 4 and idx < 24)
        samples.append(generated_sequence)
    return samples


def generate_samples_random(num_samples=100):
    samples = []
    idx_to_aa =  {v: k for k, v in esm_dict.items()}
    for i in range(num_samples):
        possible_lengths = np.arange(0, len(seq_length_freq))
        possible_aa_idx = np.arange(0, len(aa_count_freq))
        sampled_length = np.random.choice(possible_lengths, p=seq_length_freq)
        sampled_idx = np.random.choice(possible_aa_idx, size=sampled_length, p=aa_count_freq)

        sampled_sequence = ''.join(idx_to_aa[idx] for idx in sampled_idx)
        samples.append(sampled_sequence)
    return samples

def sample_with_classifier_guidance(model, classifier=None, cond_data=None,
                                    num_samples=10, max_len=14, num_classes=24,
                                    device='cpu',
                                    guidance_scale=10.0, 
                                    gradient_start_step=400,
                                    gradient_end_step = 0):
    model.eval()
    x_t = torch.randn((num_samples, max_len + 2, num_classes), device=device) # +2 for [CLS] and [EOS]
    
    progress_bar = tqdm(list(reversed(range(model.T))))
    for t in progress_bar:
        with torch.no_grad():
            t_tensor = torch.full((num_samples,), t, device=device)
            predicted_noise = model(x_t, t_tensor)
            sqrt_alpha = torch.sqrt(model.alphas[t])
            sqrt_one_minus_alpha_bar = torch.sqrt(1. - model.alpha_bars[t])
            
            # gradient guidance
            if t < gradient_start_step and t > gradient_end_step:
                x_0 = (x_t - model.betas[t]/sqrt_one_minus_alpha_bar * predicted_noise) / sqrt_alpha
                with torch.enable_grad():
                    y = x_0.detach().requires_grad_(True)
                    argmax_indices = torch.argmax(y, dim=-1)
                    y_onehot = F.one_hot(argmax_indices, num_classes=num_classes).float().requires_grad_(True)
                    classifier_score, loss = classifier.cal_score(y_onehot, cond_data)
                    gradient = torch.autograd.grad(loss, y_onehot, retain_graph=True)[0]
                    predicted_noise = predicted_noise + gradient * guidance_scale

                    progress_bar.set_postfix(loss=f'{torch.mean(loss).item():.4f}')

            x_0 = (x_t - model.betas[t]/sqrt_one_minus_alpha_bar * predicted_noise) / sqrt_alpha

            if t > 0:
                noise = torch.randn_like(x_t)
                sigma_t = torch.sqrt(model.betas[t])
                x_t = x_0 + sigma_t * noise
            else:
                x_t = x_0
         
    samples = logit_to_fasta(x_t, num_samples)
    return samples

def sample_with_classifier_guidance_resample(model, classifier=None, cond_data=None,
                                    num_samples=10, max_len=14, num_classes=24,
                                    device='cpu',
                                    guidance_scale=0.5, 
                                    gradient_start_step=400,
                                    gradient_end_step = 0,
                                    resample_times=10,
                                    resample_interval=10):
    model.eval()
    if classifier is None or cond_data is None:
        gradient_start_step = 0

    samples = []
    x_t = torch.randn((num_samples, max_len + 2, num_classes), device=device) # +2 for [CLS] and [EOS]
    
    progress_bar = tqdm(list(reversed(range(model.T))))
    for t in progress_bar:
        with torch.no_grad():
            t_tensor = torch.full((num_samples,), t, device=device)
            predicted_noise = model(x_t, t_tensor)
            sqrt_alpha = torch.sqrt(model.alphas[t])
            sqrt_one_minus_alpha_bar = torch.sqrt(1. - model.alpha_bars[t])

            x_0 = (x_t - model.betas[t]/sqrt_one_minus_alpha_bar * predicted_noise) / sqrt_alpha
            
            if t > 0:
                noise = torch.randn_like(x_t)
                sigma_t = torch.sqrt(model.betas[t])
                x_t = x_0 + sigma_t * noise
            else:
                x_t = x_0
            
            # resampling strategy with dynamically weighted gradient guidance
            if t <= gradient_start_step and t > gradient_end_step and (gradient_start_step - t) % resample_interval == 0:
                x_j = x_t
                for i in range(resample_times - 1):
                    for j in reversed(range(t - resample_interval, t)):
                        t_tensor = torch.full((num_samples,), j, device=device)
                        predicted_noise = model(x_j, t_tensor)
                        sqrt_alpha = torch.sqrt(model.alphas[j])
                        sqrt_one_minus_alpha_bar = torch.sqrt(1. - model.alpha_bars[j])
                        x_0 = (x_j - model.betas[j] / sqrt_one_minus_alpha_bar * predicted_noise) / sqrt_alpha
                        
                        with torch.enable_grad():
                            y = x_0.detach()
                            argmax_indices = torch.argmax(y, dim=-1)
                            y_onehot = F.one_hot(argmax_indices, num_classes=num_classes).float().requires_grad_(True)
                            classifier_score, loss = classifier.cal_score(y_onehot, cond_data)
                            gradient = torch.autograd.grad(loss, y_onehot, retain_graph=True)[0]
                            predicted_noise = predicted_noise + gradient * guidance_scale
                            max_total_score = classifier_score['total_score'].max().item()
                            progress_bar.set_postfix(loss=f'{torch.mean(loss).item():.4f}', score=f'{max_total_score:.4f}')

                        x_0 = (x_j - model.betas[j] / sqrt_one_minus_alpha_bar * predicted_noise) / sqrt_alpha
                        noise = torch.randn_like(x_0)
                        x_j = x_0 + torch.sqrt(model.betas[j]) * noise

                    x_j = model.add_noise_several_steps(x_j, t - resample_interval, resample_interval)
                x_t = x_j
             
    samples = logit_to_fasta(x_t, num_samples)
    return samples


if __name__ == "__main__":
    args, result_dir = parse_args()
    agpep_pred_ckpt_path = 'ckpt/agpep_pred_model.ckpt' if args.agpep else None
    speci_pred_ckpt_path = 'ckpt/speci_pred_model.ckpt' if args.specifi else None
    pmhc_pred_ckpt_path = 'ckpt/pmhc_pred_model.ckpt' if args.pmhc else None
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    diffusion_model = DiffusionModel(device=device)
    diffusion_model.load_state_dict(torch.load('./ckpt/diffusion_model.pth', map_location=device))
    diffusion_model.to(device)

    guide_classifier = ConditioningClassifier(
        agpep_pred_ckpt_path=agpep_pred_ckpt_path,
        speci_pred_ckpt_path=speci_pred_ckpt_path,
        pmhc_pred_ckpt_path=pmhc_pred_ckpt_path,
        )

    classifier = ConditioningClassifier(agpep_pred_ckpt_path='ckpt/agpep_pred_model.ckpt',
                                        speci_pred_ckpt_path='ckpt/speci_pred_model.ckpt',
                                        pmhc_pred_ckpt_path='ckpt/pmhc_pred_model.ckpt')
    
    
    test_data = pd.read_csv("data/test.csv")
    data_length = len(test_data)

    tcr_id_list = test_data['TCR_id'].tolist()
    tcr_list = test_data['TCR_seq'].tolist()
    mhc_list = test_data['MHC_name'].tolist()
    gt_pep_list = test_data['pep'].tolist()

    for idx, tcr_id, tcr_seq, mhc_name, gt_pep in zip(range(data_length), tcr_id_list, tcr_list, mhc_list, gt_pep_list):
        print(f"Generating samples for {tcr_id} ({idx+1}/{data_length})")
        cond_data = {'tcr_seq': tcr_seq, 'mhc_name': mhc_name}

        if args.mode == 'random': # random sampling
            generated_samples = generate_samples_random(num_samples=100) 
        elif args.mode == 'uncond': # unconditional generation
            generated_samples = sample_with_classifier_guidance(diffusion_model, 
                                                                num_samples=100, max_len=14, num_classes=diffusion_model.num_classes,
                                                                gradient_start_step=0, 
                                                                device=device)
        elif args.mode == 'cond': # conditional generation, without resampling
            generated_samples = sample_with_classifier_guidance(diffusion_model, guide_classifier, cond_data,
                                                                num_samples=100, max_len=14, num_classes=diffusion_model.num_classes, device=device,
                                                                gradient_start_step=400, guidance_scale=args.scale)
        elif args.mode == 'cond_resample': # conditional generation, with resampling
            generated_samples = sample_with_classifier_guidance_resample(diffusion_model, guide_classifier, cond_data,
                                                                         num_samples=100, max_len=14, num_classes=diffusion_model.num_classes, device=device,
                                                                         gradient_start_step=400, guidance_scale=args.scale,
                                                                         resample_times=10, resample_interval=10)

        classifier_result = classifier.inference_batch(generated_samples, {'tcr_seq': tcr_seq, 'mhc_name': mhc_name})
        specificity_score_list = classifier_result['specificity_score'].tolist()
        agpep_prob_list = classifier_result['agpep_prob'].tolist()
        pmhc_bind_list = classifier_result['pmhc_bind'].tolist()
        total_score_list = classifier_result['total_score'].tolist()

        result_df = pd.DataFrame({'generated_peptides': generated_samples,
                                  'specificity_score': specificity_score_list, 'agpep_prob': agpep_prob_list, 'pmhc_bind': pmhc_bind_list,
                                  'total_score': total_score_list})
        result_df = result_df.round(3)
        result_df.to_csv(os.path.join(result_dir, tcr_id+'.csv'), index=False)
        print(f"specificity_score: {round(sum(specificity_score_list)/len(specificity_score_list), 3)}, \
                agpep_prob: {round(sum(agpep_prob_list)/len(agpep_prob_list), 3)}, \
                pmhc_bind: {round(sum(pmhc_bind_list)/len(pmhc_bind_list), 3)}, \
                total_score: {round(sum(total_score_list)/len(total_score_list), 3)}")
        print(f"Saved results to {os.path.join(result_dir, tcr_id+'.csv')}")