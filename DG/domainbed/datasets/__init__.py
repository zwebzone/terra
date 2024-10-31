import torch
import numpy as np

from domainbed.datasets import datasets
from domainbed.lib import misc
from domainbed.datasets import transforms as DBT


def set_transfroms(dset, data_type, hparams, algorithm_class=None):
    """
    Args:
        data_type: ['train', 'valid', 'test', 'mnist']
    """
    assert hparams["data_augmentation"]

    additional_data = False
    if data_type == "train":
        dset.transforms = {"x": DBT.aug}
        additional_data = True
    elif data_type == "valid":
        if hparams["val_augment"] is False:
            dset.transforms = {"x": DBT.basic}
        else:
            # Originally, DomainBed use same training augmentation policy to validation.
            # We turn off the augmentation for validation as default,
            # but left the option to reproducibility.
            dset.transforms = {"x": DBT.aug}
    elif data_type == "test":
        dset.transforms = {"x": DBT.basic}
    elif data_type == "mnist":
        # No augmentation for mnist
        dset.transforms = {"x": lambda x: x}
    else:
        raise ValueError(data_type)

    if additional_data and algorithm_class is not None:
        for key, transform in algorithm_class.transforms.items():
            dset.transforms[key] = transform


# def get_dataset(test_envs, args, hparams, algorithm_class=None):
#     """Get dataset and split."""
#     is_mnist = "MNIST" in args.dataset
#     dataset = vars(datasets)[args.dataset](args.data_dir)
#     orgin_len = len(dataset)
#     #  if not isinstance(dataset, MultipleEnvironmentImageFolder):
#     #      raise ValueError("SMALL image datasets are not implemented (corrupted), for transform.")

#     # get the extra dataset for adding to the origin dataset.
#     extra_dir = args.extra_dir
    
#     if extra_dir != "None":
#         # extra_datasets = datasets.MultipleEnvironmentImageFolder(extra_dir)
#         extra_datasets = datasets.MultipleEnvironmentImageFolder2(extra_dir)
        
#         for env_i, env in enumerate(extra_datasets):
#             if env_i in args.extra_dir_index:
#                 dataset.datasets.append(env)
#                 dataset.environments.append("Extra")
#     in_splits = []
#     out_splits = []
#     for env_i, env in enumerate(dataset):
#         # The split only depends on seed_hash (= trial_seed).
#         # It means that the split is always identical only if use same trial_seed,
#         # independent to run the code where, when, or how many times.
#         out, in_ = split_dataset(
#             env,
#             int(len(env) * args.holdout_fraction),
#             misc.seed_hash(args.trial_seed, env_i),
#         )
#         if env_i in test_envs:
#             in_type = "test"
#             out_type = "test"
#         else:
#             in_type = "train"
#             out_type = "valid"

#         if is_mnist:
#             in_type = "mnist"
#             out_type = "mnist"

#         set_transfroms(in_, in_type, hparams, algorithm_class)
#         set_transfroms(out, out_type, hparams, algorithm_class)

#         if hparams["class_balanced"]:
#             in_weights = misc.make_weights_for_balanced_classes(in_)
#             out_weights = misc.make_weights_for_balanced_classes(out)
#         else:
#             in_weights, out_weights = None, None
#         in_splits.append((in_, in_weights))
#         out_splits.append((out, out_weights))

#     return dataset, in_splits, out_splits

def get_dataset(test_envs, args, hparams, algorithm_class=None):
    """Get dataset and split."""
    is_mnist = "MNIST" in args.dataset
    dataset = vars(datasets)[args.dataset](args.data_dir)
    orgin_len = len(dataset)
    #  if not isinstance(dataset, MultipleEnvironmentImageFolder):
    #      raise ValueError("SMALL image datasets are not implemented (corrupted), for transform.")

    # get the extra dataset for adding to the origin dataset.
    extra_dir = args.extra_dir
    
    if extra_dir != "None":
        extra_datasets = datasets.MultipleEnvironmentImageFolder(extra_dir)
        # extra_datasets = datasets.MultipleEnvironmentImageFolder1(extra_dir)
        
        for env_i, env in enumerate(extra_datasets):
            if env_i in args.extra_dir_index:
                dataset.datasets.append(env)
                dataset.environments.append("Extra")
    in_splits = []
    out_splits = []
    for env_i, env in enumerate(dataset):
        # The split only depends on seed_hash (= trial_seed).
        # It means that the split is always identical only if use same trial_seed,
        # independent to run the code where, when, or how many times.
        out, in_ = split_dataset(
            env,
            int(len(env) * args.holdout_fraction),
            misc.seed_hash(args.trial_seed, env_i),
        )
        if env_i in test_envs:
            in_type = "test"
            out_type = "test"
        else:
            in_type = "train"
            out_type = "valid"

        if is_mnist:
            in_type = "mnist"
            out_type = "mnist"

        set_transfroms(in_, in_type, hparams, algorithm_class)
        set_transfroms(out, out_type, hparams, algorithm_class)

        if hparams["class_balanced"]:
            in_weights = misc.make_weights_for_balanced_classes(in_)
            out_weights = misc.make_weights_for_balanced_classes(out)
        else:
            in_weights, out_weights = None, None
        in_splits.append((in_, in_weights))
        out_splits.append((out, out_weights))

    return dataset, in_splits, out_splits


class _SplitDataset(torch.utils.data.Dataset):
    """Used by split_dataset"""

    def __init__(self, underlying_dataset, keys):
        super(_SplitDataset, self).__init__()
        self.underlying_dataset = underlying_dataset
        self.keys = keys
        self.transforms = {}

        self.direct_return = isinstance(underlying_dataset, _SplitDataset)

    # def __getitem__(self, key):
    #     if self.direct_return:
    #         return self.underlying_dataset[self.keys[key]]

    #     x, y = self.underlying_dataset[self.keys[key]]

    #     ret = {"y": y}

    #     for key, transform in self.transforms.items():
    #         ret[key] = transform(x)

    #     return ret
    
    def __getitem__(self, index):
        if self.direct_return:
            # Directly return the item if the underlying dataset is already a _SplitDataset
            item = self.underlying_dataset[self.keys[index]]
            # Ensure the item is in the expected dictionary format
            if isinstance(item, tuple):
                x, y = item
                return {'x': x, 'y': y, 'path': self.underlying_dataset.underlying_dataset.samples[self.keys[index]][0]}
            return item

        # Fetch the data from the underlying dataset
        x, y = self.underlying_dataset[self.keys[index]]
        img_path = self.underlying_dataset.samples[self.keys[index]][0]  # Get image path
        # print(img_path)
        ret = {'x': x, 'y': y, 'path': img_path}  # Prepare the dictionary to return

        # Apply transforms if any
        for key, transform in self.transforms.items():
            ret[key] = transform(ret['x'])

        return ret
    
    def __len__(self):
        return len(self.keys)


def split_dataset(dataset, n, seed=0):
    """
    Return a pair of datasets corresponding to a random split of the given
    dataset, with n datapoints in the first dataset and the rest in the last,
    using the given random seed
    """
    assert n <= len(dataset)
    keys = list(range(len(dataset)))
    np.random.RandomState(seed).shuffle(keys)
    keys_1 = keys[:n]
    keys_2 = keys[n:]
    return _SplitDataset(dataset, keys_1), _SplitDataset(dataset, keys_2)
