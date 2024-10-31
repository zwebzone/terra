"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
import torch
import matplotlib

matplotlib.use('Agg')
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col


# def visualize(source_feature: torch.Tensor, target_feature: torch.Tensor,
#               filename: str, source_color='r', target_color='b'):
# def visualize(source_feature: torch.Tensor, target_feature: torch.Tensor,generated_target_feature, translated_target_feature,
#               filename: str, source_color='r', target_color='b'):
    
#     """
#     Visualize features from different domains using t-SNE.

#     Args:
#         source_feature (tensor): features from source domain in shape :math:`(minibatch, F)`
#         target_feature (tensor): features from target domain in shape :math:`(minibatch, F)`
#         filename (str): the file name to save t-SNE
#         source_color (str): the color of the source features. Default: 'r'
#         target_color (str): the color of the target features. Default: 'b'

#     """
#     source_feature = source_feature.numpy()
#     target_feature = target_feature.numpy()
#     features = np.concatenate([source_feature, target_feature], axis=0)

#     # map features to 2-d using TSNE
#     X_tsne = TSNE(n_components=2, random_state=33).fit_transform(features)

#     # domain labels, 1 represents source while 0 represents target
#     domains = np.concatenate((np.ones(len(source_feature)), np.zeros(len(target_feature))))

#     # visualize using matplotlib
#     fig, ax = plt.subplots(figsize=(10, 10))
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.spines['bottom'].set_visible(False)
#     ax.spines['left'].set_visible(False)
#     plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=domains, cmap=col.ListedColormap([target_color, source_color]), s=20)
#     plt.xticks([])
#     plt.yticks([])
#     plt.savefig(filename)

# def visualize(source_feature: torch.Tensor, target_feature: torch.Tensor, 
#               generated_target_feature: torch.Tensor, translated_target_feature: torch.Tensor,
#               filename: str, source_color='r', target_color='b', 
#               generated_target_color='g', translated_target_color='c'):
#     """
#     Visualize features from different domains using t-SNE.

#     Args:
#         source_feature (tensor): features from source domain in shape :math:`(minibatch, F)`
#         target_feature (tensor): features from target domain in shape :math:`(minibatch, F)`
#         generated_target_feature (tensor): additional features to visualize
#         translated_target_feature (tensor): additional features to visualize
#         filename (str): the file name to save t-SNE
#         source_color (str): the color of the source features. Default: 'r'
#         target_color (str): the color of the target features. Default: 'b'
#         generated_target_color (str): the color of the generated target features. Default: 'g'
#         translated_target_color (str): the color of the translated target features. Default: 'c'
#     """
#     source_feature = source_feature.numpy()
#     target_feature = target_feature.numpy()
#     generated_target_feature = generated_target_feature.numpy()
#     translated_target_feature = translated_target_feature.numpy()

#     features = np.concatenate([source_feature, target_feature, generated_target_feature, translated_target_feature], axis=0)

#     # map features to 2-d using TSNE
#     X_tsne = TSNE(n_components=2, random_state=33).fit_transform(features)

#     # domain labels for differentiating features in the plot
#     domains = np.concatenate((
#         np.full(len(source_feature), 0), 
#         np.full(len(target_feature), 1),
#         np.full(len(generated_target_feature), 2),
#         np.full(len(translated_target_feature), 3)
#     ))

#     # visualize using matplotlib
#     fig, ax = plt.subplots(figsize=(10, 10))
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.spines['bottom'].set_visible(False)
#     ax.spines['left'].set_visible(False)

#     cmap = col.ListedColormap([source_color, target_color, generated_target_color, translated_target_color])
#     scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=domains, cmap=cmap, s=20)
#     plt.legend(handles=scatter.legend_elements()[0], labels=['Source', 'Target', 'Generated Target', 'Translated Target'])
#     plt.xticks([])
#     plt.yticks([])
#     plt.savefig(filename)

from scipy.stats import gaussian_kde
# def visualize(source_feature: torch.Tensor, target_feature: torch.Tensor, 
#                                      generated_target_feature: torch.Tensor, translated_target_feature: torch.Tensor,
#                                      filename: str, source_color='r', target_color='b', 
#                                      generated_target_color='g', translated_target_color='c', dot_size=70):
#     """
#     Visualize features using t-SNE, plot target feature's contour map, and overlay dots for all features.

#     Args:
#         source_feature (tensor): features from source domain in shape (minibatch, F)
#         target_feature (tensor): features from target domain in shape (minibatch, F)
#         generated_target_feature (tensor): additional features to visualize
#         translated_target_feature (tensor): additional features to visualize
#         filename (str): the file name to save the visualization
#         source_color (str): the color of the source features. Default: 'r'
#         target_color (str): the color of the target features. Default: 'b'
#         generated_target_color (str): the color of the generated target features. Default: 'g'
#         translated_target_color (str): the color of the translated target features. Default: 'c'
#         dot_size (int): the size of the dots in the scatter plot. Default: 50
#     """
#     source_feature = source_feature.numpy()
#     target_feature = target_feature.numpy()
#     generated_target_feature = generated_target_feature.numpy()
#     translated_target_feature = translated_target_feature.numpy()

#     # Combine features for t-SNE
#     features = np.concatenate([source_feature, target_feature, generated_target_feature, translated_target_feature], axis=0)
#     X_tsne = TSNE(n_components=2, random_state=33).fit_transform(features)

#     # Extract t-SNE results for individual feature sets
#     lengths = [len(source_feature), len(target_feature), len(generated_target_feature), len(translated_target_feature)]
#     starts = np.cumsum([0] + lengths[:-1])
#     ends = np.cumsum(lengths)
#     source_tsne, target_tsne, generated_tsne, translated_tsne = [X_tsne[start:end] for start, end in zip(starts, ends)]

#     # Perform density estimation for target_feature
#     x_min, x_max = X_tsne[:, 0].min() - 1, X_tsne[:, 0].max() + 1
#     y_min, y_max = X_tsne[:, 1].min() - 1, X_tsne[:, 1].max() + 1
#     xx, yy = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
#     positions = np.vstack([xx.ravel(), yy.ravel()])
#     values = np.vstack([target_tsne[:, 0], target_tsne[:, 1]])
#     kernel = gaussian_kde(values)
#     f = np.reshape(kernel(positions).T, xx.shape)

#     fig, ax = plt.subplots(figsize=(10, 10))
#     ax.contourf(xx, yy, f, levels=15, cmap="Blues")
#     # Plot each feature set with its color and size
#     ax.scatter(source_tsne[:, 0], source_tsne[:, 1], c=source_color, s=dot_size, label='Source')
#     ax.scatter(target_tsne[:, 0], target_tsne[:, 1], c=target_color, s=dot_size, label='Target')
#     ax.scatter(generated_tsne[:, 0], generated_tsne[:, 1], c=generated_target_color, s=dot_size, label='Generated Target')
#     ax.scatter(translated_tsne[:, 0], translated_tsne[:, 1], c=translated_target_color, s=dot_size, label='Translated Target')

#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.spines['bottom'].set_visible(False)
#     ax.spines['left'].set_visible(False)
#     plt.xticks([])
#     plt.yticks([])
#     plt.legend()
#     plt.savefig(filename)

def visualize(source_feature: torch.Tensor, target_feature: torch.Tensor, 
              generated_target_feature: torch.Tensor, translated_target_feature: torch.Tensor,
              filename: str, source_color='r', target_color='royalblue', 
              generated_target_color='g', translated_target_color='gold', dot_size=90):
    """
    Visualize features using t-SNE, plot target feature's contour map, and overlay dots for all features with modified aesthetics.
    """
    source_feature = source_feature.numpy()
    target_feature = target_feature.numpy()
    generated_target_feature = generated_target_feature.numpy()
    translated_target_feature = translated_target_feature.numpy()

    # Combine features for t-SNE
    features = np.concatenate([source_feature, target_feature, generated_target_feature, translated_target_feature], axis=0)
    X_tsne = TSNE(n_components=2, random_state=33).fit_transform(features)

    # Extract t-SNE results for individual feature sets
    lengths = [len(source_feature), len(target_feature), len(generated_target_feature), len(translated_target_feature)]
    starts = np.cumsum([0] + lengths[:-1])
    ends = np.cumsum(lengths)
    source_tsne, target_tsne, generated_tsne, translated_tsne = [X_tsne[start:end] for start, end in zip(starts, ends)]

    # Perform density estimation for target_feature
    x_min, x_max = X_tsne[:, 0].min() - 1, X_tsne[:, 0].max() + 1
    y_min, y_max = X_tsne[:, 1].min() - 1, X_tsne[:, 1].max() + 1
    xx, yy = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([target_tsne[:, 0], target_tsne[:, 1]])
    kernel = gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(target_tsne[:, 0], target_tsne[:, 1], c=target_color, s=dot_size, edgecolors='white', label='Target Domain')
    ax.contourf(xx, yy, f, levels=15, cmap="Blues", alpha=0.9)

    # Adjusted target color for lower saturation and added alpha for gradient-like effect under the dots
    # adjusted_target_color = (0.7, 0.7, 1, 0.6)  # A lighter, semi-transparent blue

    # Plot each feature set with its color, size, and white edge
    ax.scatter(source_tsne[:, 0], source_tsne[:, 1], c=source_color, s=dot_size, edgecolors='white', label='Source Domain')
    ax.scatter(translated_tsne[:, 0], translated_tsne[:, 1], c=translated_target_color, s=dot_size, edgecolors='white', label='Adapted Source Domain')
    ax.scatter(generated_tsne[:, 0], generated_tsne[:, 1], c=generated_target_color, s=dot_size, edgecolors='white', label='Generated Target Domain')

    # ax.legend(fontsize=40)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.xticks([])
    plt.yticks([])
    # plt.legend()
    order = [1,0,2,3]
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],fontsize=20)

    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    plt.savefig(filename, bbox_inches='tight')




# def visualize(source_feature: torch.Tensor, target_feature: torch.Tensor, 
#                                        generated_target_feature: torch.Tensor, translated_target_feature: torch.Tensor,
#                                        filename: str, source_color='r', target_color='b', 
#                                        generated_target_color='g', translated_target_color='y', dot_size=50, line_color='y'):
#     """
#     Visualize features using t-SNE, plot target feature's contour map, overlay dots for all features, and draw lines
#     between source_feature and translated_target_feature based on matching filenames.

#     Args:
#         source_feature (tensor), target_feature (tensor), generated_target_feature (tensor), translated_target_feature (tensor): Feature tensors.
#         source_filenames (list), translated_target_filenames (list): Lists of filenames for ensuring correct correspondences.
#         filename (str): The file name to save the visualization.
#         source_color (str), target_color (str), generated_target_color (str), translated_target_color (str): Colors for each feature set.
#         dot_size (int): The size of the dots in the scatter plot. Default: 50.
#         line_color (str): Color of the lines indicating one-to-one correspondences. Default: 'y'.
#     """
#     # This assumes source_feature and translated_target_feature are already matched based on filenames.
#     # You need to implement the logic to match features based on filenames before calling this function.
    
#     # Convert tensors to NumPy arrays
#     source_feature = source_feature.numpy()
#     target_feature = target_feature.numpy()
#     generated_target_feature = generated_target_feature.numpy()
#     translated_target_feature = translated_target_feature.numpy()

#     # Combine all features for t-SNE
#     features = np.concatenate([source_feature, target_feature, generated_target_feature, translated_target_feature], axis=0)
#     X_tsne = TSNE(n_components=2, random_state=33).fit_transform(features)

#     # Extract t-SNE results for individual feature sets
#     lengths = [len(source_feature), len(target_feature), len(generated_target_feature), len(translated_target_feature)]
#     starts = np.cumsum([0] + lengths[:-1])
#     ends = np.cumsum(lengths)
#     source_tsne, target_tsne, generated_tsne, translated_tsne = [X_tsne[start:end] for start, end in zip(starts, ends)]

#     fig, ax = plt.subplots(figsize=(10, 10))

#     # Draw contour for target feature density
#     # You might need to adjust the density calculation based on your specific needs
#     x = np.linspace(min(X_tsne[:, 0]), max(X_tsne[:, 0]), 100)
#     y = np.linspace(min(X_tsne[:, 1]), max(X_tsne[:, 1]), 100)
#     X, Y = np.meshgrid(x, y)
#     positions = np.vstack([X.ravel(), Y.ravel()])
#     values = np.vstack([target_tsne[:, 0], target_tsne[:, 1]])
#     kernel = gaussian_kde(values)
#     Z = np.reshape(kernel(positions).T, X.shape)
#     ax.contourf(X, Y, Z, levels=15, cmap="Blues")

#     # Plot each feature set with its color and size
#     ax.scatter(source_tsne[:, 0], source_tsne[:, 1], c=source_color, s=dot_size, label='Source')
#     ax.scatter(target_tsne[:, 0], target_tsne[:, 1], c=target_color, s=dot_size, label='Target')
#     ax.scatter(generated_tsne[:, 0], generated_tsne[:, 1], c=generated_target_color, s=dot_size, label='Generated Target')
#     ax.scatter(translated_tsne[:, 0], translated_tsne[:, 1], c=translated_target_color, s=dot_size, label='Translated Target')

#     # Draw lines for matched source and translated target features
#     for src_pt, tgt_pt in zip(source_tsne, translated_tsne):
#         ax.plot([src_pt[0], tgt_pt[0]], [src_pt[1], tgt_pt[1]], color=line_color, linewidth=0.5)

#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.spines['bottom'].set_visible(False)
#     ax.spines['left'].set_visible(False)
#     plt.xticks([])
#     plt.yticks([])
#     plt.legend()
#     plt.savefig(filename)