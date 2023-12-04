import torch
from .dataset import valid_collate_func, PickleLoader
from .train_dataset import CustomDataset, train_collate_func


def create_valid_dataloader(ly_vocab, re_vocab, pickle_path, batch_size, num_workers):
    dataset = PickleLoader(pickle_path, ly_vocab, re_vocab, mode='test')
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        collate_fn=valid_collate_func,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )
    return dataloader


def create_train_dataloader(ly_vocab, re_vocab, train_data_path, batch_size, num_workers):
    dataset = CustomDataset(train_data_path, 'train', ly_vocab, re_vocab)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        collate_fn=train_collate_func,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )
    return dataloader


def create_test_dataloader(ly_vocab, re_vocab, pickle_path, batch_size, num_workers):
    dataset = CustomDataset(pickle_path, 'test', ly_vocab, re_vocab)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        collate_fn=train_collate_func,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )
    return dataloader


if __name__ == "__main__":
    pass
