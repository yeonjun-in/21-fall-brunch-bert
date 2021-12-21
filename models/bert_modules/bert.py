from torch import nn as nn

from models.bert_modules.embedding import BERTEmbedding
from models.bert_modules.transformer import TransformerBlock
from utils import fix_random_seed_as
import torch


class BERT(nn.Module):
    def __init__(self, args):
        super().__init__()

        fix_random_seed_as(args.model_init_seed)
        # self.init_weights()
        self.args = args
        max_len = args.bert_max_len
        num_items = args.num_items
        n_layers = args.bert_num_blocks
        heads = args.bert_num_heads
        vocab_size = num_items + 2
        hidden = args.bert_hidden_units
            
        if args.bert_emb_aggr == 'concat':

            if args.bert_add_author and not args.bert_add_magazine:
                hidden_sum = hidden*2
            elif args.bert_add_magazine and not args.bert_add_author:
                hidden_sum = hidden*2
            elif args.bert_add_author and args.bert_add_magazine:
                hidden_sum = hidden*2
            else:
                hidden_sum = hidden*1
        else:
            hidden_sum = hidden*1
        
        self.hidden = hidden
        self.hidden_sum = hidden_sum
        dropout = args.bert_dropout

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=self.hidden, max_len=max_len, dropout=dropout)
        self.embedding_m = BERTEmbedding(vocab_size=args.num_mag+2, embed_size=self.hidden, max_len=max_len, dropout=dropout)
        self.embedding_a = BERTEmbedding(vocab_size=args.num_author+2, embed_size=self.hidden, max_len=max_len, dropout=dropout)

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden_sum, heads, hidden_sum * 4, dropout) for _ in range(n_layers)])

    def forward(self, x, x_m, x_a):
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x)
        if self.args.bert_emb_aggr == 'concat':
            if self.args.bert_add_author and not self.args.bert_add_magazine:
                # print('only_add_author')
                x_a = self.embedding_a(x_a)
                x = torch.cat((x, x_a), dim=-1)
            elif self.args.bert_add_magazine and not self.args.bert_add_author:
                # print('only_add_magazine')
                x_m = self.embedding_m(x_m)
                x = torch.cat((x, x_m), dim=-1)
            elif self.args.bert_add_author and self.args.bert_add_magazine:
                # print('add_both')
                x_m = self.embedding_m(x_m)
                x_a = self.embedding_a(x_a)
                x = torch.cat((x, x_m), dim=-1)
            else:
                # print('add nothing')
                None

        if self.args.bert_emb_aggr == 'mean':
            if self.args.bert_add_author and not self.args.bert_add_magazine:
                # print('only_add_author')
                x_a = self.embedding_a(x_a)
                x = (x + x_a)/2
            elif self.args.bert_add_magazine and not self.args.bert_add_author:
                # print('only_add_magazine')
                x_m = self.embedding_m(x_m)
                x = (x + x_m)/2
            elif self.args.bert_add_author and self.args.bert_add_magazine:
                # print('add_both')
                x_m = self.embedding_m(x_m)
                x_a = self.embedding_a(x_a)
                x = (x + x_m + x_a)/3
            else:
                # print('add nothing')
                None

        if self.args.bert_emb_aggr == 'sum':
            if self.args.bert_add_author and not self.args.bert_add_magazine:
                # print('only_add_author')
                x_a = self.embedding_a(x_a)
                x = (x + x_a)
            elif self.args.bert_add_magazine and not self.args.bert_add_author:
                # print('only_add_magazine')
                x_m = self.embedding_m(x_m)
                x = (x + x_m)
            elif self.args.bert_add_author and self.args.bert_add_magazine:
                # print('add_both')
                x_m = self.embedding_m(x_m)
                x_a = self.embedding_a(x_a)
                x = (x + x_m + x_a)
            else:
                # print('add nothing')
                None
            

        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        return x

    def init_weights(self):
        pass
