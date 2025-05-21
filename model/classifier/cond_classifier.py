import json
import random
import torch
import torch.nn.functional as F

from model.classifier.agpep_pred_model import AgpepPredModel
from model.classifier.speci_pred_model import SpeciPredModel
from model.classifier.pmhc_pred_model import PMHCPredModel
from utils.constant import mhc_counts

class ConditioningClassifier():
    def __init__(self, agpep_pred_ckpt_path=None, speci_pred_ckpt_path=None, pmhc_pred_ckpt_path=None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.valid_classifier = 0
        if agpep_pred_ckpt_path != None:
            self.valid_classifier += 1
            self.agpep_pred_model = AgpepPredModel.load_from_checkpoint(agpep_pred_ckpt_path)
            self.agpep_pred_model.eval()
            self.agpep_pred_model.to(self.device)
        else: self.agpep_pred_model = None

        if speci_pred_ckpt_path != None:
            self.valid_classifier += 1
            self.speci_pred_model = SpeciPredModel.load_from_checkpoint(speci_pred_ckpt_path)
            self.speci_pred_model.eval()
            self.speci_pred_model.to(self.device)
        else: self.speci_pred_model = None

        if pmhc_pred_ckpt_path != None:
            self.valid_classifier += 1
            self.pmhc_pred_model = PMHCPredModel.load_from_checkpoint(pmhc_pred_ckpt_path)
            self.pmhc_pred_model.eval()
            self.pmhc_pred_model.to(self.device)
        else: self.pmhc_pred_model = None

        assert self.valid_classifier > 0, "Error: All classifiers are invalid! At least one classifier must be available."

        mhc_seq_file='./data/MHC_seq_dict.json'
        with open(mhc_seq_file, 'r') as f:
            self.mhc_seq_dict = json.load(f)
    

    def cal_score(self, pep_seq_onehot, conditioning_data):
        tcr_seq = conditioning_data['tcr_seq']
        mhc_name = conditioning_data['mhc_name']
        
        if mhc_name is None:
            mhc_name = random.choices(list(mhc_counts.keys()), weights=mhc_counts.values(), k=1)[0]
        mhc_seq = self.mhc_seq_dict[mhc_name]

        padding_mask = pep_seq_onehot[:, :, 1] == 1
        
        label = torch.ones(pep_seq_onehot.shape[0]).to(self.device) # bs*num_samples
        classifier_scores = []
        losses = []
        if self.speci_pred_model is not None:
            specificity = self.speci_pred_model.inference_from_onehot(tcr_seq, padding_mask, pep_seq_onehot)
            classifier_scores.append(specificity)
            losses.append(F.binary_cross_entropy(specificity, label))
        else:
            specificity = None

        if self.agpep_pred_model is not None:
            agpep_prob = self.agpep_pred_model.inference_from_onehot(padding_mask, pep_seq_onehot)
            classifier_scores.append(agpep_prob)
            losses.append(F.binary_cross_entropy(agpep_prob, label))
        else:
            agpep_prob = None

        if self.pmhc_pred_model is not None:
            pmhc_bind = self.pmhc_pred_model.inference_from_onehot(mhc_seq, padding_mask, pep_seq_onehot)
            classifier_scores.append(pmhc_bind)
            losses.append(F.binary_cross_entropy(pmhc_bind, label))
        else:
            pmhc_bind = None

        loss = sum(losses)
        total_score = sum(classifier_scores) / self.valid_classifier
        
        classifier_score = {"specificity_score": specificity.detach() if specificity is not None else None,
                            "agpep_prob": agpep_prob.detach() if agpep_prob is not None else None,
                            "pmhc_bind": pmhc_bind.detach() if pmhc_bind is not None else None,
                            "total_score": total_score.detach()}
        return classifier_score, loss
    
    @torch.no_grad()
    def inference_batch(self, pep_seq_list, conditioning_data):
        tcr_seq = conditioning_data['tcr_seq']
        mhc_name = conditioning_data['mhc_name']
        
        if mhc_name is None:
            mhc_name = random.choices(list(mhc_counts.keys()), weights=mhc_counts.values(), k=1)[0]
        mhc_seq = self.mhc_seq_dict[mhc_name]

        classifier_scores = []
        if self.speci_pred_model is not None:
            specificity = self.speci_pred_model.inference_batch(tcr_seq, pep_seq_list)
            classifier_scores.append(specificity)
        else:
            specificity = None

        if self.agpep_pred_model is not None:
            agpep_prob = self.agpep_pred_model.inference_batch(pep_seq_list)
            classifier_scores.append(agpep_prob)
        else:
            agpep_prob = None

        if self.pmhc_pred_model is not None:
            pmhc_bind = self.pmhc_pred_model.inference_batch(mhc_seq, pep_seq_list)
            classifier_scores.append(pmhc_bind)
        else:
            pmhc_bind = None

        total_score = sum(classifier_scores) / self.valid_classifier
        
        classifier_score = {"specificity_score": specificity.detach() if specificity is not None else None,
                            "agpep_prob": agpep_prob.detach() if agpep_prob is not None else None,
                            "pmhc_bind": pmhc_bind.detach() if pmhc_bind is not None else None,
                            "total_score": total_score.detach()}
        return classifier_score