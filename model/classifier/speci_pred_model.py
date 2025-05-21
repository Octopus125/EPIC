# TCR-p specificity prediction model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import AUROC, AveragePrecision
import pytorch_lightning as pl
import esm

from utils.classifier import MLPClassifier
from utils.esm_tools import esm_forward2

class SpeciPredModel(pl.LightningModule):
    def __init__(self, input_dim=640, hidden_dim=256, num_classes=2,
                 learning_rate=1e-4):
        super().__init__()

        self.esm_model, self.alphabet = esm.pretrained.esm2_t6_8M_UR50D()
        self.batch_converter = self.alphabet.get_batch_converter()
        self.esm_model.train()

        self.classifier = MLPClassifier(input_dim, hidden_dim, num_classes)
        self.classification_loss = nn.CrossEntropyLoss()
        self.auroc = AUROC(task='binary')
        self.aupr = AveragePrecision(task='binary')

        self.learning_rate = learning_rate


    def forward(self, batch):
        batch_size = len(batch['label'])
        tcr_seq_data = [(x, seq) for x, seq in zip(range(batch_size), batch['tcr_seq'])]
        pep_seq_data = [(x, seq) for x, seq in zip(range(batch_size), batch['pep_seq'])]

        _, _, tcr_batch_tokens = self.batch_converter(tcr_seq_data)
        tcr_batch_tokens = tcr_batch_tokens.to(self.device)
        tcr_batch_lens = (tcr_batch_tokens != self.alphabet.padding_idx).sum(1)

        results = self.esm_model(tcr_batch_tokens, repr_layers=[6], return_contacts=True)
        tcr_token_representations = results["representations"][6]

        _, _, pep_batch_tokens = self.batch_converter(pep_seq_data)
        pep_batch_tokens = pep_batch_tokens.to(self.device)
        pep_batch_lens = (pep_batch_tokens != self.alphabet.padding_idx).sum(1)

        results = self.esm_model(pep_batch_tokens, repr_layers=[6], return_contacts=True)
        pep_token_representations = results["representations"][6]

        sequence_representations = []
        for i in range(batch_size):
            tcr_token_len = tcr_batch_lens[i]
            pep_token_len = pep_batch_lens[i]
            tcr_repre = tcr_token_representations[i, 1 : tcr_token_len - 1].mean(0)
            pep_repre = pep_token_representations[i, 1 : pep_token_len - 1].mean(0)
            sequence_representations.append(torch.cat((tcr_repre, pep_repre)))
        sequence_representations = torch.stack(sequence_representations)
        
        output = self.classifier(sequence_representations)
        return output

    def training_step(self, batch, batch_idx):
        seq_pred = self.forward(batch)
        loss = self.classification_loss(seq_pred, batch['label'])
        self.log("train/loss", loss)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        seq_pred = self.forward(batch)
        val_loss = self.classification_loss(seq_pred, batch['label'])
        self.log("valid_loss", val_loss)
        probabilities = F.softmax(seq_pred, dim=1)
        positive_prob = probabilities[:, 1]
        self.auroc.update(positive_prob, batch['label'])
        self.aupr.update(positive_prob, batch['label'])
        return {"val_loss": val_loss}
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.mean(torch.stack([x["val_loss"] for x in outputs]))
        self.log("valid_loss", avg_loss)

        auroc = self.auroc.compute()
        self.log('val_auroc', auroc)
        self.auroc.reset()

        aupr = self.aupr.compute()
        self.log('val_aupr', aupr)
        self.aupr.reset()
    

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([
            # sequence
            {'params': self.esm_model.parameters(), 'lr': self.learning_rate},
            {'params': self.classifier.parameters(), 'lr': self.learning_rate},
        ])
        return optimizer
    
    @torch.no_grad()
    def inference(self, tcr_seq, pep_seq):
        tcr_seq_data = [('tcr', tcr_seq)]
        pep_seq_data = [('pep', pep_seq)]

        _, _, tcr_batch_tokens = self.batch_converter(tcr_seq_data)
        tcr_batch_tokens = tcr_batch_tokens.to(self.device)
        tcr_batch_lens = (tcr_batch_tokens != self.alphabet.padding_idx).sum(1)

        results = self.esm_model(tcr_batch_tokens, repr_layers=[6], return_contacts=True)
        tcr_token_representations = results["representations"][6]

        _, _, pep_batch_tokens = self.batch_converter(pep_seq_data)
        pep_batch_tokens = pep_batch_tokens.to(self.device)
        pep_batch_lens = (pep_batch_tokens != self.alphabet.padding_idx).sum(1)

        results = self.esm_model(pep_batch_tokens, repr_layers=[6], return_contacts=True)
        pep_token_representations = results["representations"][6]

        tcr_token_len = tcr_batch_lens[0]
        pep_token_len = pep_batch_lens[0]
        tcr_repre = tcr_token_representations[0, 1 : tcr_token_len - 1].mean(0)
        pep_repre = pep_token_representations[0, 1 : pep_token_len - 1].mean(0)
        sequence_representation = torch.cat((tcr_repre, pep_repre))
        
        output = self.classifier(sequence_representation)
        probabilities = F.softmax(output, dim=0)
        positive_prob = probabilities[1]
        return positive_prob
    
    @torch.no_grad()
    def inference_batch(self, tcr_seq, pep_seq_list):
        batch_size = len(pep_seq_list)
        tcr_seq_data = [('tcr', tcr_seq)]
        pep_seq_data = [(x, seq) for x, seq in zip(range(batch_size), pep_seq_list)]

        _, _, tcr_batch_tokens = self.batch_converter(tcr_seq_data)
        tcr_batch_tokens = tcr_batch_tokens.to(self.device)
        tcr_batch_lens = (tcr_batch_tokens != self.alphabet.padding_idx).sum(1)

        results = self.esm_model(tcr_batch_tokens, repr_layers=[6], return_contacts=True)
        tcr_token_representations = results["representations"][6]

        _, _, pep_batch_tokens = self.batch_converter(pep_seq_data)
        pep_batch_tokens = pep_batch_tokens.to(self.device)
        pep_batch_lens = (pep_batch_tokens != self.alphabet.padding_idx).sum(1)

        results = self.esm_model(pep_batch_tokens, repr_layers=[6], return_contacts=True)
        pep_token_representations = results["representations"][6]

        sequence_representations = []
        tcr_token_len = tcr_batch_lens[0]
        tcr_repre = tcr_token_representations[0, 1 : tcr_token_len - 1].mean(0)
        for i in range(batch_size):
            pep_token_len = pep_batch_lens[i]
            pep_repre = pep_token_representations[i, 1 : pep_token_len - 1].mean(0)
            sequence_representations.append(torch.cat((tcr_repre, pep_repre)))
        sequence_representations = torch.stack(sequence_representations)
        
        output = self.classifier(sequence_representations)
        probabilities = F.softmax(output, dim=1)
        positive_prob = probabilities[:, 1]
        return positive_prob
    

    def inference_from_onehot(self, tcr_seq, padding_mask, pep_token_onehot): # for classifier guidance
        assert padding_mask.shape[0] == pep_token_onehot.shape[0]

        tcr_seq_data = [('tcr', tcr_seq)]
        _, _, tcr_batch_tokens = self.batch_converter(tcr_seq_data)
        tcr_batch_tokens = tcr_batch_tokens.to(self.device)
        tcr_batch_lens = (tcr_batch_tokens != self.alphabet.padding_idx).sum(1)
        results = self.esm_model(tcr_batch_tokens, repr_layers=[6], return_contacts=True)
        tcr_token_representations = results["representations"][6] # [1, tcr_len, 320]

        pep_batch_lens = (pep_token_onehot[:, :, 1] != 1).sum(1) # [bs]
        esm_embed_layer_weight = self.esm_model.embed_tokens.weight
        pep_embedding = torch.matmul(pep_token_onehot, esm_embed_layer_weight[:pep_token_onehot.shape[-1], :]) # (bs*num_samples, max_len, 320)
        results = esm_forward2(self.esm_model, padding_mask, pep_embedding, repr_layers=[6])
        pep_token_representations = results["representations"][6] #[bs, len+2, 320]

        sequence_representations = []
        tcr_token_len = tcr_batch_lens[0]
        tcr_repre = tcr_token_representations[0, 1 : tcr_token_len - 1].mean(0) # [320]
        for i, pep_token_len in enumerate(pep_batch_lens):
            pep_repre = pep_token_representations[i, 1 : pep_token_len - 1].mean(0) # [320]
            sequence_representations.append(torch.cat((tcr_repre, pep_repre)))
        sequence_representations = torch.stack(sequence_representations) # [bs*num_sample,640]
 
        output = self.classifier(sequence_representations)
        probabilities = F.softmax(output, dim=1)
        positive_prob = probabilities[:, 1]
        return positive_prob

