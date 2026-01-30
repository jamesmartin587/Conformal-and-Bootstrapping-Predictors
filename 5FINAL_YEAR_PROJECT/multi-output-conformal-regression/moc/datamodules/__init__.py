from .real_datamodule import RealDataModule
from .toy_datamodule import ToyDataModule
from .cifar10_datamodule import CIFAR10DataModule
from .stocks_datamodule import BootstrapDataModule
from moc.configs.datasets import toy_dataset_groups, real_dataset_groups


def get_datamodule(group):
    if group in toy_dataset_groups:
        return ToyDataModule
    elif group in real_dataset_groups:
        return RealDataModule
    elif group == 'cifar10':
        return CIFAR10DataModule
    elif group == 'stocks':
        return RealDataModule
    raise ValueError(f'Unknown datamodule {group}')

def get_bootstrap_datamodule(group):
    if group == 'stocks':
        return BootstrapDataModule
    raise ValueError(f'Unknown datamodule {group}')

def load_datamodule(rc):
    datamodule_cls = get_datamodule(rc.dataset_group)
    return datamodule_cls(
        rc=rc,
        seed=2000 + rc.run_id,
    )

def bootstrap_load_datamodule(rc):
    datamodule_cls = get_bootstrap_datamodule(rc.dataset_group)
    return datamodule_cls(
        rc=rc,
        seed=2000 + rc.run_id,
    )    
