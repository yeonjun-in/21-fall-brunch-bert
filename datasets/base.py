from .utils import *
from config import RAW_DATASET_ROOT_FOLDER

import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

from abc import *
from pathlib import Path
import os
import tempfile
import shutil
import pickle


class AbstractDataset(metaclass=ABCMeta):
    def __init__(self, args):
        self.args = args
        self.min_rating = args.min_rating
        self.min_uc = args.min_uc
        self.min_sc = args.min_sc
        self.split = args.split

        assert self.min_uc >= 2, 'Need at least 2 ratings per user for validation and test'

    @classmethod
    @abstractmethod
    def code(cls):
        pass

    @classmethod
    def raw_code(cls):
        return cls.code()
    
    @classmethod
    def all_raw_file_names(cls):
        return []

    @abstractmethod
    def load_ratings_df(self):
        pass

    def load_dataset(self):
        self.preprocess()
        dataset_path = self._get_preprocessed_dataset_path()
        dataset = pickle.load(dataset_path.open('rb'))
        return dataset

    def preprocess(self):
        dataset_path = self._get_preprocessed_dataset_path()
        if dataset_path.is_file():
            print('Already preprocessed. Skip preprocessing')
            path = '/'.join(str(dataset_path).split('/')[:-1])
            self.args.input_path = path
            return
        if not dataset_path.parent.is_dir():
            dataset_path.parent.mkdir(parents=True)
            
        df = self.load_ratings_df()
        df = self.make_implicit(df)
        df = self.filter_triplets(df)
        df, umap, smap, amap, mmap = self.densify_index(df)
        item_freq = df['sid'].value_counts().reset_index().rename(columns={'index':'sid', 'sid':'freq'})
        train, train_m, train_a, val, val_m, val_a, test, test_m, test_a = self.split_df(df, len(umap))
        dataset = {'train': train,
                   'train_m': train_m,
                   'train_a': train_a, 
                   'val': val,
                   'val_m': val_m,
                   'val_a': val_a,
                   'test_m': test_m,
                   'test_a': test_a,
                   'test': test,
                   'umap': umap,
                   'smap': smap,
                   'amap': amap,
                   'mmap': mmap,
                   }
        with dataset_path.open('wb') as f:
            pickle.dump(dataset, f)
        
        path = '/'.join(str(dataset_path).split('/')[:-1])
        self.args.input_path = path+'/item_freq.csv'
        item_freq.to_csv(path+'/item_freq.csv', index=False)
            
    def make_implicit(self, df):
        print('Turning into implicit ratings')
        df = df[df['rating'] >= self.min_rating]
        # return df[['uid', 'sid', 'timestamp']]
        return df

    def filter_triplets(self, df):
        print('Filtering triplets')
        if self.min_sc > 0:
            item_sizes = df.groupby('sid').size()
            good_items = item_sizes.index[item_sizes >= self.min_sc]
            df = df[df['sid'].isin(good_items)]

        if self.min_uc > 0:
            user_sizes = df.groupby('uid').size()
            good_users = user_sizes.index[user_sizes >= self.min_uc]
            df = df[df['uid'].isin(good_users)]

        return df

    def densify_index(self, df):
        print('Densifying index')
        umap = {u: i for i, u in enumerate(set(df['uid']))}
        smap = {s: i for i, s in enumerate(set(df['sid']))}
        amap = {a: i for i, a in enumerate(set(df['user_id']))}
        mmap = {a: i for i, a in enumerate(set(df['magazine_id']))}
        
        df['uid'] = df['uid'].map(umap)
        df['sid'] = df['sid'].map(smap)
        df['user_id'] = df['user_id'].map(amap)
        df['magazine_id'] = df['magazine_id'].map(mmap)
        return df, umap, smap, amap, mmap

    def split_df(self, df, user_count):
        if self.args.split == 'leave_one_out':
            print('Splitting')
            user_group = df.groupby('uid')
            user2items = user_group.progress_apply(lambda d: list(d.sort_values(by='timestamp')['sid']))
            user2mags = user_group.progress_apply(lambda d: list(d.sort_values(by='timestamp')['magazine_id']))
            user2auths = user_group.progress_apply(lambda d: list(d.sort_values(by='timestamp')['user_id']))
            train, train_m, train_a, val, val_m, val_a, test, test_m, test_a = {}, {}, {}, {}, {}, {}, {}, {}, {}
            for user in range(user_count):
                items = user2items[user]
                mags = user2mags[user]
                auth = user2auths[user]
                train[user], val[user], test[user] = items[:-2], items[-2:-1], items[-1:]
                train_m[user], val_m[user], test_m[user] = mags[:-2], mags[-2:-1], mags[-1:]
                train_a[user], val_a[user], test_a[user] = auth[:-2], auth[-2:-1], auth[-1:]
            return train, train_m, train_a, val, val_m, val_a, test, test_m, test_a
        elif self.args.split == 'holdout':
            print('Splitting')
            np.random.seed(self.args.dataset_split_seed)
            eval_set_size = self.args.eval_set_size

            # Generate user indices
            permuted_index = np.random.permutation(user_count)
            train_user_index = permuted_index[                :-2*eval_set_size]
            val_user_index   = permuted_index[-2*eval_set_size:  -eval_set_size]
            test_user_index  = permuted_index[  -eval_set_size:                ]

            # Split DataFrames
            train_df = df.loc[df['uid'].isin(train_user_index)]
            val_df   = df.loc[df['uid'].isin(val_user_index)]
            test_df  = df.loc[df['uid'].isin(test_user_index)]

            # DataFrame to dict => {uid : list of sid's}
            train = dict(train_df.groupby('uid').progress_apply(lambda d: list(d['sid'])))
            val   = dict(val_df.groupby('uid').progress_apply(lambda d: list(d['sid'])))
            test  = dict(test_df.groupby('uid').progress_apply(lambda d: list(d['sid'])))
            return train, val, test
        else:
            raise NotImplementedError

    def _get_rawdata_root_path(self):
        return Path(RAW_DATASET_ROOT_FOLDER)

    def _get_rawdata_folder_path(self):
        root = self._get_rawdata_root_path()
        return root.joinpath(self.raw_code())

    def _get_preprocessed_root_path(self):
        root = self._get_rawdata_root_path()
        return root.joinpath('preprocessed')

    def _get_preprocessed_folder_path(self):
        preprocessed_root = self._get_preprocessed_root_path()
        folder_name = '{}_min_rating{}-min_uc{}-min_sc{}-split{}' \
            .format(self.code(), self.min_rating, self.min_uc, self.min_sc, self.split)
        return preprocessed_root.joinpath(folder_name)

    def _get_preprocessed_dataset_path(self):
        folder = self._get_preprocessed_folder_path()
        return folder.joinpath('dataset.pkl')

