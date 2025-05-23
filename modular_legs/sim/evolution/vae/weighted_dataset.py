import argparse
import os
from pathlib import Path
import pdb
import pickle
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import pytorch_lightning as pl

from modular_legs import LEG_ROOT_DIR
from modular_legs.sim.evolution.encoding_wrapper import to_onehot
from modular_legs.sim.evolution.vae import utils

NUM_WORKERS = 3


class WeightedDataset(pl.LightningDataModule):
    """ Implements a weighted numpy dataset (used for shapes task) """

    def __init__(self, hparams):
        super().__init__()
        self.dataset_path = hparams.dataset_path
        self.cfg = hparams
        self.val_frac = hparams.val_frac
        self.property_key = hparams.property_key
        self.batch_size = hparams.batch_size
        

    def prepare_data(self):
        pass

    # @staticmethod
    # def _get_tensor_dataset(data):
    #     data = torch.as_tensor(data, dtype=torch.long)
    #     data = torch.nn.functional.one_hot(data, num_classes=-1)
    #     data = torch.permute(data, (0,3,1,2)).float()
    #     # data = torch.unsqueeze(data, 1)
    #     return TensorDataset(data)

    def setup(self, stage):
        if self.dataset_path.endswith(".npz"):
            raise NotImplementedError("This is not implemented yet.")
            with np.load(self.dataset_path) as npz:
                all_data = npz["data"]
                if self.property_key in npz:
                    all_properties = npz[self.property_key]
                else:
                    all_properties = np.ones(all_data.shape[0])
                    self.cfg.rank_weight_k = np.inf
                    print("This is a unweighted pre-training.")
        elif self.dataset_path.endswith(".pkl"):
            with open(self.dataset_path, "rb") as f:
                all_data = pickle.load(f)
                self.sampled_original_data = random.sample(all_data, 10) # For investigating the VAE
                assert isinstance(all_data, list), "Data must be a list."
                self.max_idx = max(max(sublist) for sublist in all_data)
                self.max_length = max(len(sublist) for sublist in all_data)
                print("max_idx: ", self.max_idx)
                all_data = to_onehot(all_data, self.max_idx, self.max_length)
                all_properties = np.ones(all_data.shape[0])
                self.cfg.rank_weight_k = np.inf
                print("This is a unweighted pre-training.")
        assert all_properties.shape[0] == all_data.shape[0]
        self.data_shape = all_data.shape


        N_val = int(all_data.shape[0] * self.val_frac)
        self.data_val = all_data[:N_val]
        self.prop_val = all_properties[:N_val]
        self.data_train = all_data[N_val:]
        self.prop_train = all_properties[N_val:]

        # Make into tensor datasets
        self.train_dataset = TensorDataset(self.data_train)
        self.val_dataset = TensorDataset(self.data_val)

        self.data_weighter = utils.DataWeighter(self.cfg)
        self.set_weights()

    def set_weights(self):
        """ sets the weights from the weighted dataset """

        # Make train/val weights
        self.train_weights = self.data_weighter.weighting_function(self.prop_train)
        self.val_weights = self.data_weighter.weighting_function(self.prop_val)

        # Create samplers
        self.train_sampler = WeightedRandomSampler(
            self.train_weights, num_samples=len(self.train_weights), replacement=True
        )
        self.val_sampler = WeightedRandomSampler(
            self.val_weights, num_samples=len(self.val_weights), replacement=True
        )

    def append_train_data(self, x_new, prop_new):

        # Special adjustment for fb-vae: only add the best points
        if self.data_weighter.weight_type == "fb":

            # Find top quantile
            cutoff = np.quantile(prop_new, self.data_weighter.weight_quantile)
            indices_to_add = prop_new >= cutoff

            # Filter all but top quantile
            x_new = x_new[indices_to_add]
            prop_new = prop_new[indices_to_add]
            assert len(x_new) == len(prop_new)

            # Replace data (assuming that number of samples taken is less than the dataset size)
            self.data_train = np.concatenate(
                [self.data_train[len(x_new) :], x_new], axis=0
            )
            self.prop_train = np.concatenate(
                [self.prop_train[len(x_new) :], prop_new], axis=0
            )
        else:

            # Normal treatment: just concatenate the points
            self.data_train = np.concatenate([self.data_train, x_new], axis=0)
            self.prop_train = np.concatenate([self.prop_train, prop_new], axis=0)
        self.train_dataset = WeightedNumpyDataset._get_tensor_dataset(self.data_train)
        self.set_weights()

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=NUM_WORKERS,
            sampler=self.train_sampler,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=NUM_WORKERS,
            sampler=self.val_sampler,
            drop_last=True,
        )
