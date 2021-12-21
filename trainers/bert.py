from .base import AbstractTrainer
from .utils import recalls_and_ndcgs_for_ks

import torch.nn as nn
import torch


class BERTTrainer(AbstractTrainer):
    def __init__(self, args, model, train_loader, val_loader, test_loader, export_root):
        super().__init__(args, model, train_loader, val_loader, test_loader, export_root)
        self.ce = nn.CrossEntropyLoss(ignore_index=0)

    @classmethod
    def code(cls):
        return 'bert'

    def add_extra_loggers(self):
        pass

    def log_extra_train_info(self, log_data):
        pass

    def log_extra_val_info(self, log_data):
        pass

    def calculate_loss(self, batch):
        seqs, seq_m, seq_a, labels = batch
        logits = self.model(seqs, seq_m, seq_a)  # B x T x V

        logits = logits.view(-1, logits.size(-1))  # (B*T) x V
        labels = labels.view(-1)  # B*T
        loss = self.ce(logits, labels)
        ## 여기서 cross entropy 가 아니라 negative sample 을 이용한 bpr loss 로 학습이 되도록 해야겠군
        return loss

    def calculate_metrics(self, batch):
        seqs, seqs_m, seqs_a, candidates, labels = batch
        scores = self.model(seqs, seqs_m, seqs_a)  # B x T x V
        scores = scores[:, -1, :]  # B x V
        
        topk_item = scores.topk(max(self.metric_ks), dim=1)[1]
        tailK = {}
        for k in sorted(self.metric_ks, reverse=True):
            items = topk_item[:, :k]
            tailK[f'Tail@{k}'] = (torch.isin(items, self.tail_item.unsqueeze(0)).sum(1) / k).mean()
            
        scores = scores.gather(1, candidates)  # B x C
        metrics = recalls_and_ndcgs_for_ks(scores, labels, self.metric_ks)
        for k,v in tailK.items():
            metrics[k] = v.item()
        return metrics

    def only_inference(self, batch, k):
        seqs, seq_m, seq_a = batch[0]
        scores = self.model(seqs, seq_m, seq_a)  # B x T x V
        scores = scores[:, -1, :]  # B x V
        topk_item = scores.topk(k, dim=1)
        return topk_item[1].detach().cpu().numpy()

