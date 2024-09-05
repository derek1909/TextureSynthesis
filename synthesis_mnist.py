'''
This module is a Python implementation of:

    A. Efros and T. Leung, "Texture Synthesis by Non-parametric Sampling,"
    Proceedings of the Seventh IEEE International Conference on Computer
    Vision, September 1999.

Specifically, this module implements texture synthesis by growing a 3x3 texture patch 
pixel-by-pixel. Please see the authors' project page for additional algorithm details: 

    https://people.eecs.berkeley.edu/~efros/research/EfrosLeung.html

Example:

    Generate a 50x50 texture patch from a texture available at the input path and save it to
    the output path. Also, visualize the synthesis process:

        $ python synthesis.py --sample_path=[input path] --out_path=[output path] --visualize

'''

__author__ = 'Derek Dong'

# import argparse
# import cv2
# import numpy as np
import time
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Check if CUDA is available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

EIGHT_CONNECTED_NEIGHBOR_KERNEL = torch.tensor([[1., 1., 1.],
                                               [1., 0., 1.],
                                               [1., 1., 1.]], dtype=torch.float64, device=device)
SIGMA_COEFF = 6.4      # The denominator for a 2D Gaussian sigma used in the reference implementation.
ERROR_THRESHOLD = 0.1  # The default error threshold for synthesis acceptance in the reference implementation.


def find_normalized_ssd(sample, window, mask, window_index):
    # Get the kernel size and create the Gaussian kernel
    kernel_size, _ = window.shape
    _, _, sh, sw = sample.shape

    # Form a 2D Gaussian weight matrix from symmetric linearly separable Gaussian kernels
    sigma = kernel_size / SIGMA_COEFF
    x = torch.arange(kernel_size, device=device, dtype=torch.float64)
    kernel_1d = torch.exp(- (x-(kernel_size-1)/2)**2 / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()  # Normalize the kernel
    kernel_2d = kernel_1d[:, None] @ kernel_1d[None, :] * mask
    kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)  # add batch and channel dimension for conv2d
    # plt.figure()
    # plt.imshow(sample[2000,0].cpu())

    # Define the padding area
    pad = 2
    context = kernel_size // 2  # Context based on kernel size
    # print('pad=',pad, ', context=', context)
    # Calculate indices with context for kernel
    start_row_conv = max(0, window_index[0] - pad - context)
    end_row_conv = min(sh, window_index[0] + pad + context + 1)
    start_col_conv = max(0, window_index[1] - pad - context)
    end_col_conv = min(sw, window_index[1] + pad + context + 1)

    # print('window index: ', window_index[0],window_index[1] )
    # print('conv index: ', start_row_conv,end_row_conv,start_col_conv,end_col_conv)

    # Extract the subsample area
    subsample = sample[..., start_row_conv:end_row_conv, start_col_conv:end_col_conv]

    # Apply convolution on the subsample
    local_ssd = F.conv2d(subsample**2, kernel_2d, padding='same') - 2 * F.conv2d(subsample, kernel_2d * window, padding='same') + (window**2 * kernel_2d).sum()

    # Initialize a full size SSD tensor
    ssd_full = torch.full(sample.shape, float('inf'), device=device, dtype=torch.float64)
    ssd_full[..., start_row_conv:end_row_conv, start_col_conv:end_col_conv] = local_ssd  # squeeze if necessary depending on your channel dimension setup

    # Trim local_ssd
    start_row = max(0, window_index[0] - context)
    end_row = min(sh, window_index[0] + context + 1)
    start_col = max(0, window_index[1] - context)
    end_col = min(sw, window_index[1] + context + 1)
    # plt.figure()
    # plt.imshow(ssd_full[2000,0].cpu())

    # Trim ssd_full (within convolution range but above the target window)
    ssd_full[..., start_row_conv:start_row, start_col_conv:end_col_conv] = float('inf') # Top rows 
    ssd_full[..., end_row:end_row_conv, start_col_conv:end_col_conv] = float('inf') # Bottom rows
    ssd_full[..., start_row_conv:end_row_conv, start_col_conv:start_col] = float('inf') # Left columns
    ssd_full[..., start_row_conv:end_row_conv, end_col:end_col_conv] = float('inf') # Right columns
    
    # print('ssd_full',ssd_full.shape)
    # Normalize the SSD
    normalize_factor = kernel_2d.sum().item()
    normalized_ssd = torch.maximum(torch.tensor(0.0, device=device), ssd_full / normalize_factor)
    # plt.figure()
    # plt.imshow(ssd_full[2000,0].cpu())
    # error
    return normalized_ssd

def get_candidate_indices(normalized_ssd, error_threshold=ERROR_THRESHOLD):
    min_non_zero_ssd = normalized_ssd[normalized_ssd > 0].min() # Get the minimum non-zero SSD value
    min_threshold = min_non_zero_ssd * (1. + error_threshold)
    indices = torch.nonzero(normalized_ssd <= min_threshold, as_tuple=True)
    # print('min_non_zero_ssd',min_non_zero_ssd)
    # print('min_threshold',min_threshold)
    # print('indices',indices.shape)
    
    # indices = (tensor containing indices of first axis, tensor containing indices of second axis, tensor containing indices of third axis, tensor containing indices of forth axis)
    return indices

def select_pixel_index(normalized_ssd, indices, method='uniform'):
    N = indices[0].shape[0]
    # print('num of selection pool', N)

    if method == 'uniform':
        weights = torch.ones(N) / N
        selection = torch.randint(0, N, (1,)).item()
        # selection = torch.multinomial(weights, 1).item()

    else:
        # this option does work now - due to ssd might be zero (clipped to zero from negative value)
        weights = normalized_ssd[indices]
        weights = weights / torch.sum(weights)
        selection = torch.multinomial(weights, 1).item()

    # accccf = normalized_ssd[indices]
    # print(accccf.shape)

    # Select a random pixel index based on weights
    # print('selection', selection)
    # print()
    selected_index = tuple(index[selection] for index in indices)
    # print('selected_index',selected_index)
    # print('indices[:][selection]',indices[:][selection])
    # print('tuple(index[selection] for index in indices)',tuple(index[selection] for index in indices))

    return selected_index

def get_neighboring_pixel_indices(pixel_mask):
    # Taking the difference between the dilated mask and the initial mask
    # gives only the 8-connected neighbors of the mask frontier.
    kernel = torch.ones((3, 3), dtype=torch.float64, device=device)
    dilated_mask = F.conv2d(pixel_mask.unsqueeze(0).unsqueeze(0), 
                            kernel.unsqueeze(0).unsqueeze(0), padding='same') 
    dilated_mask = (dilated_mask > 0).float() # make it binary
    dilated_mask = dilated_mask.squeeze(0).squeeze(0)

    neighbors = dilated_mask - pixel_mask

    # Recover the indices of the mask frontier.
    neighbor_indices = torch.nonzero(neighbors)

    return neighbor_indices

def permute_neighbors(pixel_mask, neighbors):
    # neighbors: (N,2)

    # Generate a permutation of the neighboring indices
    permuted_indices = torch.randperm(neighbors.shape[0])
    permuted_neighbors = neighbors[permuted_indices,:]


    # Use convolution to count the number of existing neighbors for all entries in the mask.
    neighbor_count = F.conv2d(pixel_mask.unsqueeze(0).unsqueeze(0), 
                              EIGHT_CONNECTED_NEIGHBOR_KERNEL.unsqueeze(0).unsqueeze(0), padding='same')
    neighbor_count = neighbor_count.squeeze().squeeze()

    # print('neighbor_count',neighbor_count.shape)
    # Sort the permuted neighboring indices by quantity of existing neighbors descending.
    permuted_neighbor_counts = neighbor_count[permuted_neighbors[:, 0], permuted_neighbors[:, 1]]

    # print('permuted_neighbors1',permuted_neighbors)

    sorted_order = torch.argsort(permuted_neighbor_counts, descending=True)
    # print('sorted_order',sorted_order)
    permuted_neighbors = permuted_neighbors[sorted_order,:]
    # print('permuted_neighbors2',permuted_neighbors)

    return permuted_neighbors

def texture_can_be_synthesized(mask):
    # The texture can be synthesized while the mask has unfilled entries.
    mh, mw = mask.shape[:2]
    num_completed = torch.sum(mask != 0).item()
    num_incomplete = (mh * mw) - num_completed
    
    return num_incomplete > 0

def initialize_texture_synthesis(test_sample, test_label, window_size, kernel_size, seed_size=3):

    # Generate window and mask
    window = torch.zeros(window_size, dtype=torch.float64, device=device)
    mask = torch.zeros(window_size, dtype=torch.float64, device=device)

    # Initialize window with random seed from sample
    # seed = a random 3x3 patch
    sx, sy, sh, sw = test_sample.shape
    ix = torch.randint(0, sx, (1,))
    iy = torch.randint(0, sy, (1,))
    # ih = torch.randint(0, sh-seed_size+1, (1,))
    # iw = torch.randint(0, sw-seed_size+1, (1,))
    ih, iw = (sh - seed_size + 1) // 2, (sw - seed_size + 1) // 2

    # select a central seed
    # ih, iw = (window_size[0] // 2) - 1, (window_size[1] // 2) - 1

    seed = test_sample[ix, iy, ih:ih+seed_size, iw:iw+seed_size]
    selected_label = test_label[ix].item()
    original_image = test_sample[ix, iy, :, :]
    plt.figure()
    plt.imshow(original_image.squeeze().cpu(), vmin=0, vmax=1, cmap='grey')
    plt.title(f'original image: {selected_label}')

    # Place seed in center of window
    ph, pw = (window_size[0] - seed_size + 1) // 2, (window_size[1] - seed_size + 1) // 2
    window[ph:ph+seed_size, pw:pw+seed_size] = seed
    mask[ph:ph+seed_size, pw:pw+seed_size] = 1

    # Obtain padded versions of window and mask
    pad = kernel_size // 2
    padded_window = torch.nn.functional.pad(window, pad=(pad, pad, pad, pad), mode='constant', value=0)
    padded_mask = torch.nn.functional.pad(mask, pad=(pad, pad, pad, pad), mode='constant', value=0)
    # Obtain views of the padded window and mask
    window = padded_window[pad:-pad, pad:-pad]
    mask = padded_mask[pad:-pad, pad:-pad]
    return window, selected_label, mask, padded_window, padded_mask

def filter_sample(label, selected_label, sample):
    """
    Filters the sample to keep only the images with the specified label.

    Parameters:
    - label: The label to filter by (e.g., 0, 1, ..., 9).
    - sample: A tuple containing (images, labels).
        - images: A tensor of shape (N, C, H, W) where N is the number of images.
        - labels: A tensor of shape (N,) where N is the number of labels.

    Returns:
    - filtered_images: A tensor containing images with the specified label.
    - filtered_labels: A tensor containing only the specified label.
    """

    # Create a boolean mask where True corresponds to the given label
    mask = (label == selected_label)

    # import ipdb; ipdb.set_trace()

    # Filter the images and labels using the mask
    filtered_sample = sample[mask]
    filtered_labels = label[mask]

    # print('filtered_sample: ', filtered_sample.shape)
    return filtered_sample, filtered_labels

def synthesize_texture(sample, label, test_sample, test_label, window_size, kernel_size, seed_size):

    start_time = time.time()

    sample = sample.to(dtype=torch.float64,device=device)
    
    (window, selected_label, mask, padded_window, padded_mask) = initialize_texture_synthesis(test_sample, test_label, window_size, kernel_size, seed_size)

    filtered_sample, filtered_labels = filter_sample(label, selected_label, sample)
    # end_time = time.time()
    # execution_time = end_time - start_time
    # print(f'Initialization finished. Time used: {execution_time:.1f}s')

    # Synthesize texture until all pixels in the window are filled.
    while texture_can_be_synthesized(mask):
        # Get neighboring indices that are neighbors of the already synthesized pixels
        neighboring_indices = get_neighboring_pixel_indices(mask)

        # Permute and sort neighboring indices by number of sythesised pixels in 8-connected neighbors.
        neighboring_indices = permute_neighbors(mask, neighboring_indices)
        
        for i in range(neighboring_indices.shape[0]):
            ch, cw = neighboring_indices[i]
            window_slice = padded_window[ch:ch+kernel_size, cw:cw+kernel_size]
            mask_slice = padded_mask[ch:ch+kernel_size, cw:cw+kernel_size]
            
            # Translate index to accommodate padding.
            window_index = (ch + kernel_size // 2, cw + kernel_size // 2)

            # Compute SSD for the current pixel neighborhood and select an index with low error.
            ssd = find_normalized_ssd(filtered_sample, window_slice, mask_slice, window_index)
            # print('ssd', ssd.shape)
            indices = get_candidate_indices(ssd)
            # print('incides', indices)
            selected_index = select_pixel_index(ssd, indices)
            # print('selected_index2', selected_index)

            # Translate index to accommodate padding.
            # selected_index = (selected_index[0], selected_index[1], selected_index[2] + kernel_size // 2, selected_index[3] + kernel_size // 2)
            # print('selected_index3',selected_index)

            # print('selected_index', selected_index)
            # Set windows and mask.
            # This will update padded_window and padded_mask as well
            window[ch, cw] = filtered_sample[selected_index]
            mask[ch, cw] = 1

    #         if visualize:
    #             cv2.imshow('synthesis window', result_window)
    #             key = cv2.waitKey(1) 
    #             if key == 27:
    #                 cv2.destroyAllWindows()
    #                 return result_window

    # if visualize:
    #     cv2.imshow('synthesis window', result_window)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    end_time = time.time()
    execution_time = end_time - start_time

    print(f'Synthesis finished. Time used: {execution_time:.1f}s')
    return window