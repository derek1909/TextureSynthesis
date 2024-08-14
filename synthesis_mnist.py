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

import argparse
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


def find_normalized_ssd(sample, window, mask):
    # Get the kernel size and create the Gaussian kernel
    wh, ww = window.shape
    # Form a 2D Gaussian weight matrix from symmetric linearly separable Gaussian kernels
    sigma = wh / SIGMA_COEFF
    x = torch.arange(wh, device=device, dtype=torch.float64)
    kernel_1d = torch.exp(- (x-(wh-1)/2)**2 / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()  # Normalize the kernel
    kernel_2d = kernel_1d[:, None] @ kernel_1d[None, :] * mask

    # Apply convolution to compute SSD
    # (a-b)^2 = a^2+b^2-2ab
    window = window.unsqueeze(0).unsqueeze(0)
    kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)
    ssd= F.conv2d(sample**2, kernel_2d) - 2 * F.conv2d(sample, kernel_2d * window) + (window**2 * kernel_2d).sum()
    # Normalize the SSD
    normalize_factor = kernel_2d.sum().item()
    normalized_ssd = torch.maximum(torch.tensor(0.0), ssd / normalize_factor)

    # plt.imshow(normalized_ssd.squeeze().squeeze().cpu().numpy())
    # error
    return normalized_ssd

def get_candidate_indices(normalized_ssd, error_threshold=ERROR_THRESHOLD):
    min_non_zero_ssd = normalized_ssd[normalized_ssd > 0].min() # Get the minimum non-zero SSD value
    min_threshold = min_non_zero_ssd * (1. + error_threshold)
    indices = torch.nonzero(normalized_ssd <= min_threshold, as_tuple=True)
    return indices

def select_pixel_index(normalized_ssd, indices, method='uniform'):
    N = indices[0].shape[0]

    if method == 'uniform':
        weights = torch.ones(N) / N
    else:
        # this option does work now - due to ssd might be zero (clipped to zero from negative value)
        weights = normalized_ssd[indices]
        weights = weights / torch.sum(weights)

    # Select a random pixel index based on weights
    selection = torch.multinomial(weights, 1).item()
    selected_index = (indices[0][selection], indices[1][selection], indices[2][selection], indices[3][selection])
    
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

def initialize_texture_synthesis(sample, window_size, kernel_size):

    sample = sample.to(dtype=torch.float64,device=device) # (BatchSize, Channels, H, W) = (1, 1, 530, 530)

    # Generate working window, output window and mask
    window = torch.zeros(window_size, dtype=torch.float64, device=device)
    result_window = torch.zeros_like(window, dtype=torch.uint8, device=device)
    mask = torch.zeros(window_size, dtype=torch.float64, device=device)

    # Initialize window with random seed from sample
    # seed = a random 3x3 patch
    sx, sy, sh, sw = sample.shape
    ix = torch.randint(0, sx, (1,))
    iy = torch.randint(0, sy, (1,))
    # ih = torch.randint(0, sh-3+1, (1,))
    # iw = torch.randint(0, sw-3+1, (1,))

    ih = 12
    iw = 12
    seed = sample[ix, iy, ih:ih+3, iw:iw+3]

    # Place seed in center of window
    ph, pw = (window_size[0] // 2) - 1, (window_size[1] // 2) - 1
    window[ph:ph+3, pw:pw+3] = seed
    mask[ph:ph+3, pw:pw+3] = 1
    result_window[ph:ph+3, pw:pw+3] = sample[ix, iy, ih:ih+3, iw:iw+3]
    # print('original_sample[ih:ih+3, iw:iw+3]',original_sample[ih:ih+3, iw:iw+3].shape)

    # error
    # Obtain padded versions of window and mask
    pad = kernel_size // 2
    padded_window = torch.nn.functional.pad(window, pad=(pad, pad, pad, pad), mode='constant', value=0)
    padded_mask = torch.nn.functional.pad(mask, pad=(pad, pad, pad, pad), mode='constant', value=0)
    # Obtain views of the padded window and mask
    window = padded_window[pad:-pad, pad:-pad]
    mask = padded_mask[pad:-pad, pad:-pad]

    return sample, window, mask, padded_window, padded_mask, result_window
    
def synthesize_texture(original_sample, window_size, kernel_size, visualize):

    start_time = time.time()
    
    (sample, window, mask, padded_window, 
        padded_mask, result_window) = initialize_texture_synthesis(original_sample, window_size, kernel_size)

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

            # Compute SSD for the current pixel neighborhood and select an index with low error.
            ssd = find_normalized_ssd(sample, window_slice, mask_slice)
            # print('ssd', ssd.shape)
            indices = get_candidate_indices(ssd)
            # print('incides', indices)
            selected_index = select_pixel_index(ssd, indices)
            # print('selected_index', selected_index)

            # Translate index to accommodate padding.
            selected_index = (selected_index[0], selected_index[1], selected_index[2] + kernel_size // 2, selected_index[3] + kernel_size // 2)
            # print('selected_index', selected_index)
            # Set windows and mask.
            # This will update padded_window and padded_mask as well
            window[ch, cw] = sample[selected_index]
            mask[ch, cw] = 1
            result_window[ch, cw] = original_sample[selected_index[0], selected_index[1], selected_index[2], selected_index[3]]

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
    return result_window