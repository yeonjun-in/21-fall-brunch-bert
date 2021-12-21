from abc import *
import random


class AbstractDataloader(metaclass=ABCMeta):
    def __init__(self, args, dataset):
        self.args = args
        seed = args.dataloader_random_seed
        self.rng = random.Random(seed)
        self.save_folder = dataset._get_preprocessed_folder_path()
        dataset = dataset.load_dataset()
        self.train = dataset['train']
        self.train_m = dataset['train_m']
        self.train_a = dataset['train_a']
        self.val = dataset['val']
        self.val_m = dataset['val_m']
        self.val_a = dataset['val_a']
        self.test = dataset['test']
        self.test_m = dataset['test_m']
        self.test_a = dataset['test_a']
        self.umap = dataset['umap']
        self.smap = dataset['smap']
        self.amap = dataset['amap']
        self.mmap = dataset['mmap']
        self.user_count = len(self.umap)
        self.item_count = len(self.smap)
        self.mag_count = len(self.mmap)
        self.auth_count = len(self.amap)

    @classmethod
    @abstractmethod
    def code(cls):
        pass

    @abstractmethod
    def get_pytorch_dataloaders(self):
        pass
