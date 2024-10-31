import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import tqdm


# def collect_feature(data_loader: DataLoader, feature_extractor: nn.Module,
#                     device: torch.device, max_num_features=None) -> torch.Tensor:
#     """
#     Fetch data from `data_loader`, and then use `feature_extractor` to collect features

#     Args:
#         data_loader (torch.utils.data.DataLoader): Data loader.
#         feature_extractor (torch.nn.Module): A feature extractor.
#         device (torch.device)
#         max_num_features (int): The max number of features to return

#     Returns:
#         Features in shape (min(len(data_loader), max_num_features * mini-batch size), :math:`|\mathcal{F}|`).
#     """
#     feature_extractor.eval()
#     all_features = []
#     with torch.no_grad():
#         for i, (images, target) in enumerate(tqdm.tqdm(data_loader)):
#             if max_num_features is not None and i >= max_num_features:
#                 break
#             images = images.to(device)
#             feature = feature_extractor(images).cpu()
#             all_features.append(feature)
#     return torch.cat(all_features, dim=0)

def collect_feature(data_loader: torch.utils.data.DataLoader, feature_extractor: torch.nn.Module,
                    device: torch.device, return_labels: bool = False, max_num_features=None) -> tuple:
    """
    Fetch data from `data_loader`, and then use `feature_extractor` to collect features. Optionally return labels.

    Args:
        data_loader (torch.utils.data.DataLoader): Data loader.
        feature_extractor (torch.nn.Module): A feature extractor.
        device (torch.device): Device to run the feature extraction on.
        return_labels (bool): If True, returns a tuple of features and labels. Default is False.
        max_num_features (int, optional): The max number of features to return. If None, return all features.

    Returns:
        If return_labels is False, returns features in shape 
        (min(len(data_loader), max_num_features * mini-batch size), :math:`|\mathcal{F}|`).
        If return_labels is True, returns a tuple (features, labels), where labels are the corresponding labels
        of the features.
    """
    feature_extractor.eval()
    all_features = []
    all_labels = []  # Prepare to collect labels

    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm.tqdm(data_loader)):
            if max_num_features is not None and len(all_features) >= max_num_features:
                break
            images = images.to(device)
            features = feature_extractor(images).cpu()
            all_features.append(features)
            if return_labels:
                all_labels.append(labels.cpu())  # Collect labels

    features = torch.cat(all_features, dim=0)
    if return_labels:
        labels = torch.cat(all_labels, dim=0)
        return features, labels
    return features
