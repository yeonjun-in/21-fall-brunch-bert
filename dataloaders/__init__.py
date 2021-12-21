from datasets import dataset_factory
from .bert import BertDataloader
from .ae import AEDataloader


DATALOADERS = {
    BertDataloader.code(): BertDataloader,
    AEDataloader.code(): AEDataloader
}


def dataloader_factory(args):
    dataset = dataset_factory(args)
    dataloader = DATALOADERS[args.dataloader_code]
    dataloader = dataloader(args, dataset)
    train, val, test = dataloader.get_pytorch_dataloaders()
    args.num_author = len(dataset.load_dataset()['amap'])
    args.num_mag = len(dataset.load_dataset()['mmap'])
    return train, val, test
