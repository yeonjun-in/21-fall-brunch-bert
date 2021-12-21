from .base import AbstractDataloader
from .negative_samplers import negative_sampler_factory
from .data_augmentation import Crop, Mask, Reorder, Substitute, Insert, Random, CombinatorialEnumerate

import torch
import torch.utils.data as data_utils
import random
import numpy as np

import random
import copy
import itertools
import pandas as pd
import os, pickle

class OfflineItemSimilarity:
    def __init__(self, data_file=None, similarity_path=None, model_name='ItemCF', \
        dataset_name='Sports_and_Outdoors'):
        self.similarity_path = similarity_path
        self.model_name = model_name
        self.similarity_model = self.load_similarity_model(self.similarity_path)
        self.max_score, self.min_score = self.get_maximum_minimum_sim_scores()
        
    def get_maximum_minimum_sim_scores(self):
        max_score, min_score = -1, 100
        for item in self.similarity_model.keys():
            for neig in self.similarity_model[item]:
                sim_score = self.similarity_model[item][neig]
                max_score = max(max_score, sim_score)
                min_score = min(min_score, sim_score)
        return max_score, min_score
    
    def _convert_data_to_dict(self, data):
        """
        split the data set
        testdata is a test data set
        traindata is a train set
        """
        train_data_dict = {}
        for user,item,record in data:
            train_data_dict.setdefault(user,{})
            train_data_dict[user][item] = record
        return train_data_dict

    def _save_dict(self, dict_data, save_path = './similarity.pkl'):
        print("saving data to ", save_path)
        with open(save_path, 'wb') as write_file:
            pickle.dump(dict_data, write_file)

    def load_similarity_model(self, similarity_model_path):
        if not similarity_model_path:
            raise ValueError('invalid path')
        elif not os.path.exists(similarity_model_path):
            print("the similirity dict not exist, generating...")
            self._generate_item_similarity(save_path=self.similarity_path)
        if self.model_name in ['ItemCF', 'ItemCF_IUF', 'Item2Vec', 'LightGCN', 'Tag2Vec']:
            with open(similarity_model_path, 'rb') as read_file:
                similarity_dict = pickle.load(read_file)
            return similarity_dict
        elif self.model_name == 'Random':
            similarity_dict = self.train_item_list
            return similarity_dict

    def most_similar(self, item, top_k=1, with_score=False):
        if self.model_name in ['ItemCF', 'ItemCF_IUF', 'Item2Vec', 'LightGCN', 'Tag2Vec']:
            """TODO: handle case that item not in keys"""
            if str(item) in self.similarity_model:
                top_k_items_with_score = sorted(self.similarity_model[str(item)].items(),key=lambda x : x[1], \
                                            reverse=True)[0:top_k]
                if with_score:
                    return list(map(lambda x: (int(x[0]), (self.max_score - float(x[1]))/(self.max_score -self.min_score)), top_k_items_with_score))
                return list(map(lambda x: int(x[0]), top_k_items_with_score))
            elif int(item) in self.similarity_model:
                top_k_items_with_score = sorted(self.similarity_model[int(item)].items(),key=lambda x : x[1], \
                                            reverse=True)[0:top_k]
                if with_score:
                    return list(map(lambda x: (int(x[0]), (self.max_score - float(x[1]))/(self.max_score -self.min_score)), top_k_items_with_score))
                return list(map(lambda x: int(x[0]), top_k_items_with_score))
            else:
                item_list = list(self.similarity_model.keys())
                random_items = random.sample(item_list, k=top_k)
                if with_score:
                    return list(map(lambda x: (int(x), 0.0), random_items))
                return list(map(lambda x: int(x), random_items))
        elif self.model_name == 'Random':
            random_items = random.sample(self.similarity_model, k = top_k)
            if with_score:
                return list(map(lambda x: (int(x), 0.0), random_items))
            return list(map(lambda x: int(x), random_items))

class BertDataloader(AbstractDataloader):
    def __init__(self, args, dataset):
        super().__init__(args, dataset)
        args.num_items = len(self.smap)
        self.max_len = args.bert_max_len
        self.mask_prob = args.bert_mask_prob
        self.CLOZE_MASK_TOKEN = self.item_count + 1

        code = args.train_negative_sampler_code
        train_negative_sampler = negative_sampler_factory(code, self.train, self.val, self.test,
                                                          self.user_count, self.item_count,
                                                          args.train_negative_sample_size,
                                                          args.train_negative_sampling_seed,
                                                          self.save_folder)
        code = args.test_negative_sampler_code
        test_negative_sampler = negative_sampler_factory(code, self.train, self.val, self.test,
                                                         self.user_count, self.item_count,
                                                         args.test_negative_sample_size,
                                                         args.test_negative_sampling_seed,
                                                         self.save_folder)

        self.train_negative_samples = train_negative_sampler.get_negative_samples()
        self.test_negative_samples = test_negative_sampler.get_negative_samples()

    @classmethod
    def code(cls):
        return 'bert'

    def get_pytorch_dataloaders(self):
        train_loader = self._get_train_loader()
        val_loader = self._get_val_loader()
        test_loader = self._get_test_loader()
        return train_loader, val_loader, test_loader

    def _get_train_loader(self):
        dataset = self._get_train_dataset()
        dataloader = data_utils.DataLoader(dataset, batch_size=self.args.train_batch_size,
                                           shuffle=True, pin_memory=True)
        return dataloader

    def _get_train_dataset(self):
        dataset = BertTrainDataset(self.train, self.train_m, self.train_a, self.max_len, self.mask_prob, self.CLOZE_MASK_TOKEN, self.item_count, self.rng, self.args)
        return dataset

    def _get_val_loader(self):
        return self._get_eval_loader(mode='val')

    def _get_test_loader(self):
        return self._get_eval_loader(mode='test')

    def _get_eval_loader(self, mode):
        batch_size = self.args.val_batch_size if mode == 'val' else self.args.test_batch_size
        dataset = self._get_eval_dataset(mode)
        dataloader = data_utils.DataLoader(dataset, batch_size=batch_size,
                                           shuffle=False, pin_memory=True)
        return dataloader

    def _get_eval_dataset(self, mode):
        answers = self.val if mode == 'val' else self.test
        dataset = BertEvalDataset(self.train, self.train_m, self.train_a, answers, self.max_len, self.CLOZE_MASK_TOKEN, self.test_negative_samples)
        return dataset


class BertTrainDataset(data_utils.Dataset):
    def __init__(self, u2seq, u2seq_m, u2seq_a, max_len, mask_prob, mask_token, num_items, rng, args):
        self.u2seq = u2seq
        self.u2seq_m = u2seq_m
        self.u2seq_a = u2seq_a
        self.users = sorted(self.u2seq.keys())
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.mask_token = mask_token
        self.mask_token_m = len(u2seq_m)+1
        self.mask_token_a = len(u2seq_a)+1
        self.num_items = num_items
        self.num_mags = len(u2seq_m)
        self.num_auths = len(u2seq_a)
        self.rng = rng
        self.args = args
        self.aug_list = self.args.bert_aug_list
        self.df = pd.read_csv("df.csv")
        self.mmap = dict(self.df[['sid', 'magazine_id']].drop_duplicates().values)
        self.amap = dict(self.df[['sid', 'user_id']].drop_duplicates().values)
        self.similarity_model = OfflineItemSimilarity(similarity_path='Tag2Vec.pkl', model_name='Tag2Vec')
        
        self.augmentations = {'crop': Crop(tao=0.9),
                              'mask': Mask(gamma=0.1),
                              'reorder': Reorder(beta=0.3),
                            }
        
        
        self.augmentations = {'crop': Crop(tao=0.9),
                              'mask': Mask(gamma=0.1),
                              'reorder': Reorder(beta=0.3),
                              'substitute': Substitute(self.similarity_model,
                                                substitute_rate=0.3),
                              'insert': Insert(self.similarity_model, 
                                               insert_rate=0.3,
                                               max_insert_num_per_pos=3)
                            }        
        
        '''
        self.augmentations = {'crop': Crop(tao=0.9),
                              'mask': Mask(gamma=0.1),
                              'reorder': Reorder(beta=0.3),
                              'substitute': Substitute(self.similarity_model,
                                                substitute_rate=args.substitute_rate),
                              'insert': Insert(self.similarity_model, 
                                               insert_rate=args.insert_rate,
                                               max_insert_num_per_pos=args.max_insert_num_per_pos),
                              'random': Random(tao=args.tao, gamma=args.gamma, 
                                                beta=args.beta, item_similarity_model=self.similarity_model,
                                                insert_rate=args.insert_rate, 
                                                max_insert_num_per_pos=args.max_insert_num_per_pos,
                                                substitute_rate=args.substitute_rate,
                                                augment_threshold=self.args.augment_threshold,
                                                augment_type_for_short=self.args.augment_type_for_short),
                              'combinatorial_enumerate': CombinatorialEnumerate(tao=args.tao, gamma=args.gamma, 
                                                beta=args.beta, item_similarity_model=self.similarity_model,
                                                insert_rate=args.insert_rate, 
                                                max_insert_num_per_pos=args.max_insert_num_per_pos,
                                                substitute_rate=args.substitute_rate, n_views=args.n_views)
                            }
        '''
    def __len__(self):
        return len(self.users)
    
    def __getitem__(self, index):
        user = self.users[index]
        seq = self._getseq(user)
        
        p = np.random.uniform(0, 1)
        if self.aug_list is not None and p < 0.5:
            # print('aug')
            i = random.randint(0, len(self.aug_list)-1)
            aug_f = self.augmentations[self.aug_list[i]]
            seq = aug_f(seq)
            
        seq_m = [self.mmap[i] for i in seq]
        seq_a = [self.amap[i] for i in seq]
        
        tokens = []
        tokens_m = []
        tokens_a = []
        labels = []
        for s, m, a in zip(seq, seq_m, seq_a):
            prob = self.rng.random()
            if prob < self.mask_prob:
                prob /= self.mask_prob

                if prob < 0.8:
                    tokens.append(self.mask_token)
                    tokens_m.append(self.mask_token_m)
                    tokens_a.append(self.mask_token_a)
                elif prob < 0.9:
                    tokens.append(self.rng.randint(1, self.num_items))
                    tokens_m.append(self.rng.randint(1, self.num_mags))
                    tokens_a.append(self.rng.randint(1, self.num_auths))
                else:
                    tokens.append(s)
                    tokens_m.append(m)
                    tokens_a.append(a)

                labels.append(s)
            else:
                tokens.append(s)
                tokens_m.append(m)
                tokens_a.append(a)
                labels.append(0)

        tokens = tokens[-self.max_len:]
        tokens_m = tokens_m[-self.max_len:]
        tokens_a = tokens_a[-self.max_len:]
        labels = labels[-self.max_len:]

        mask_len = self.max_len - len(tokens)

        tokens = [0] * mask_len + tokens
        tokens_m = [0] * mask_len + tokens_m
        tokens_a = [0] * mask_len + tokens_a
        labels = [0] * mask_len + labels

        return torch.LongTensor(tokens), torch.LongTensor(tokens_m), torch.LongTensor(tokens_a), torch.LongTensor(labels)

    def _getseq(self, user):
        return self.u2seq[user]



class BertEvalDataset(data_utils.Dataset):
    def __init__(self, u2seq, u2seq_m, u2seq_a, u2answer, max_len, mask_token, negative_samples):
        self.u2seq = u2seq
        self.u2seq_m = u2seq_m
        self.u2seq_a = u2seq_a
        
        self.users = sorted(self.u2seq.keys())
        self.u2answer = u2answer
        self.max_len = max_len
        self.mask_token = mask_token
        self.mask_token_m = len(u2seq_m)+1
        self.mask_token_a = len(u2seq_a)+1
        self.negative_samples = negative_samples

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self.u2seq[user]
        seq_m = self.u2seq_m[user]
        seq_a = self.u2seq_a[user]

        answer = self.u2answer[user]
        negs = self.negative_samples[user]

        candidates = answer + negs
        labels = [1] * len(answer) + [0] * len(negs)

        seq = seq + [self.mask_token]
        seq = seq[-self.max_len:]
        padding_len = self.max_len - len(seq)
        seq = [0] * padding_len + seq

        seq_m = seq_m + [self.mask_token_m]
        seq_m = seq_m[-self.max_len:]
        padding_len = self.max_len - len(seq_m)
        seq_m = [0] * padding_len + seq_m

        seq_a = seq_a + [self.mask_token_a]
        seq_a = seq_a[-self.max_len:]
        padding_len = self.max_len - len(seq_a)
        seq_a = [0] * padding_len + seq_a

        return torch.LongTensor(seq), torch.LongTensor(seq_m), torch.LongTensor(seq_a), torch.LongTensor(candidates), torch.LongTensor(labels)

class BertInferDataset(data_utils.Dataset):
    def __init__(self, u2seq, u2seq_m, u2seq_a,  max_len, mask_token, negative_samples):
        self.u2seq = u2seq
        self.u2seq_m = u2seq_m
        self.u2seq_a = u2seq_a

        self.users = sorted(self.u2seq.keys())
        self.max_len = max_len
        self.mask_token = mask_token
        self.mask_token_m = len(u2seq_m)+1
        self.mask_token_a = len(u2seq_a)+1
        self.negative_samples = negative_samples

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self.u2seq[user]
        seq_m = self.u2seq_m[user]
        seq_a = self.u2seq_a[user]

        seq = seq + [self.mask_token]
        seq = seq[-self.max_len:]
        padding_len = self.max_len - len(seq)
        seq = [0] * padding_len + seq

        seq_m = seq_m + [self.mask_token_m]
        seq_m = seq_m[-self.max_len:]
        padding_len = self.max_len - len(seq_m)
        seq_m = [0] * padding_len + seq_m

        seq_a = seq_a + [self.mask_token_a]
        seq_a = seq_a[-self.max_len:]
        padding_len = self.max_len - len(seq_a)
        seq_a = [0] * padding_len + seq_a

        return [torch.LongTensor(seq), torch.LongTensor(seq_m), torch.LongTensor(seq_a)]

