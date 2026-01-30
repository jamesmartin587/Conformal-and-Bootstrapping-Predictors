from .preprocessing import stock_preprocess
from pathlib import Path
import logging
import pandas as pd
import numpy as np
import torch
from .base_datamodule import BaseDataModule
from torch.utils.data.dataset import Subset
from .base_datamodule import StandardScaler
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
log = logging.getLogger('moc')
import itertools

def get_data_stocks(data_dir, name):
    nb_targets_dict = {
        'NVDA2025-8-17-10':1,
        'NVDA2025-8-20-10':1,
        'NVDA2025-8-20-20':1,
        'NVDA2025-8-27-20':1,
        #ADD NEW DATASETS HERE
    }

    data_path = Path(data_dir)
    path = data_path / 'stocks' / f'{name}.csv'
    df = pd.read_csv(path)
    nb_targets = nb_targets_dict[name]
    x = df.iloc[:,:-nb_targets]
    y = df.iloc[:, -nb_targets:]
    x,y = stock_preprocess(x,y)
    return x,y

def get_data(data_dir, group, name):
    if group == 'stocks':
        return get_data_stocks(data_dir, name)
    else:
        raise ValueError(f'Unknown group: {group}')


class BootstrapDataModule(BaseDataModule):
    def get_data(self):
        return get_data(self.rc.config.data_dir, self.dataset_group, self.dataset)
    
    def load_datasets(self):
        x, y = self.get_data()
        x = torch.from_numpy(x).to(torch.float32)
        y = torch.from_numpy(y).to(torch.float32)
        tensor_data = TensorDataset(x, y)
        self.total_size = len(tensor_data)
        # Convert ratios to number of elements in the dataset
        t,v,c,ts = self.train_val_calib_test_split_ratio
        self.train_val_calib_test_split_ratio = [t,v+ts,c,0]
        splits_size = np.array(self.train_val_calib_test_split_ratio) * len(tensor_data)
        log.debug(f'Total size: {len(tensor_data)}')
        log.debug(f'Splits size before: {splits_size}')
        calib_index = 2
        to_remove_from_calib = max(0, splits_size[calib_index] - 2048)
        splits_size[calib_index] -= to_remove_from_calib
        # Don't add points in splits that should be empty (e.g., interleaving is often empty)
        mask = (splits_size != 0) & (np.arange(len(splits_size)) != calib_index)
        splits_size[mask] += to_remove_from_calib / mask.sum()
        splits_size = splits_size.astype(int)
        splits_size[-1] = len(tensor_data) - splits_size[:-1].sum()
        log.debug(f'Splits size after: {splits_size}')
   
        splits_list = splits_size.tolist()
        indices = [i for i in range(x.shape[0])]
        print("tensors",tensor_data)
        (self.data_train,self.data_val,self.data_calib,self.data_test,) = [Subset(tensor_data, indices[offset - length : offset])
        for offset, length in zip(itertools.accumulate(splits_list), splits_list)]
        x, y = self.data_train[:]
        self.scaler_x = StandardScaler().fit(x)
        self.scaler_y = StandardScaler().fit(y)
        if self.rc.config.normalize:
            self.data_train = self.make_scaled_dataset(self.data_train)
            self.data_val = self.make_scaled_dataset(self.data_val)
            self.data_calib = self.make_scaled_dataset(self.data_calib)
            self.data_test = self.make_scaled_dataset(self.data_test)
  
        # Make the size of the inputs accessible to the models
        first_x, first_y = self.data_train[0]
        self.input_dim = first_x.shape[0]
        self.output_dim = first_y.shape[0]
        
    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=len(self.data_train),
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            drop_last=False,
        )
