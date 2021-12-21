from .base import AbstractDataset

import pandas as pd
from datetime import date
import os, sys
import tqdm

def iterate_data_files(from_dtm, to_dtm):
    from_dtm, to_dtm = map(str, [from_dtm, to_dtm])
    read_root = os.path.join('./', 'read')
    for fname in os.listdir(read_root):
        if len(fname) != len('2018100100_2018100103'):
            continue
        if from_dtm != 'None' and from_dtm > fname:
            continue
        if to_dtm != 'None' and fname > to_dtm:
            continue
        path = os.path.join(read_root, fname)
        yield path, fname

class BrunchDataset(AbstractDataset):
    @classmethod
    def code(cls):
        return 'brunch'
    
    @classmethod
    def url(cls):
        return 'http://https://arena.kakao.com/c/6/data/read.tar'
    
    def load_ratings_df(self):
        data = []
        files = sorted([path for path, _ in iterate_data_files('2018100100', '2019022200')])
        for path in tqdm.tqdm(files, mininterval=1):
            for line in open(path):
                tokens = line.strip().split()
                read_datetime = path[7:17]
                user_id = tokens[0]
                reads = tokens[1:]
                for item in reads:
                    data.append([read_datetime, user_id, item])

        import pandas as pd

        read_df = pd.DataFrame(data)
        read_df.columns = ['timestamp', 'uid', 'sid']
        read_df['rating'] = 1
        read_df = read_df.drop_duplicates().reset_index(drop=True)
        
        magazine = pd.read_json('meta/magazine.json', lines=True)
        metadata = pd.read_json('meta/metadata.json', lines=True)
        users = pd.read_json('meta/users.json', lines=True)
        
        read_df = read_df.merge(metadata[['magazine_id', 'user_id', 'id']], left_on='sid', right_on='id', how='left').dropna(0)

        # tmp = read_df.groupby('uid')['timestamp'].last()

        users = list(pd.read_csv('./predict/dev.users', header=None)[0].values)
        users += list(pd.read_csv('./predict/test.users', header=None)[0].values)
        # users += tmp[tmp >= '2019010100'].index.tolist()
        users = list(set(users))
        read_df = read_df[read_df['uid'].isin(users)].reset_index(drop=True) # dev랑 test 에 있는 user 만으로 모델링 ... 이게 맞나... 싶은데 일단 keep going
        
        return read_df
        