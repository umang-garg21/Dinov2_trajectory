# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import torch
import random


def collate_data_and_cast(samples_list, mask_ratio_tuple, mins_per_slot, mask_probability, dtype, n_tokens=None, mask_generator=None):
    # dtype = torch.half  # TODO: Remove

    n_global_crops = len(samples_list[0][0]["global_crops"])
    n_local_crops = len(samples_list[0][0]["local_crops"])
    token_w = mins_per_slot

    collated_global_crops = torch.stack([s[0]["global_crops"][i] for i in range(n_global_crops) for s in samples_list])
    collated_local_crops = torch.stack([s[0]["local_crops"][i] for i in range(n_local_crops) for s in samples_list])

    B = len(collated_global_crops)  # 64 x2 (batchsize x global_crops_per_img)
    N = n_tokens  ## Total number of tokens that are crafted out of global crop

    n_samples_masked = int(B * mask_probability)   # 128x0.5 = 64 samples 
    probs = torch.linspace(*mask_ratio_tuple, n_samples_masked + 1)  # how many tokens to be masked for different sample.
    upperbound = 0
    masks_list = []

    # Total global crop tensor shaped = 64x2x144x240
    # 50% global crops are masked. => 50% selected images used => only 50% tokens masked. 
    # Out of these 50
    for i in range(0, n_samples_masked): # select 50% global crops
        prob_min = probs[i]
        prob_max = probs[i + 1]

        # Pass the number of tokens should be masked out 144x240 (should be less than 0.5x144x240)
        masks_list.append(torch.BoolTensor(mask_generator(int(N * random.uniform(prob_min, prob_max))))) # MASK_LIST[0]=144x240
        upperbound += int(N* prob_max) # flattened upper bound of the actual masked tokens index # upperbound[0]=0.1x144, ......, upperbound[63]=0.5x144
    for i in range(n_samples_masked, B):   # Fill the rest 50%crops with all FALSE.
        masks_list.append(torch.BoolTensor(mask_generator(0)))

    random.shuffle(masks_list)

    temp = torch.stack(masks_list)
    all_same_mask = torch.all(temp == temp[:, :, 0:1], dim=2)  # Shape: [C, H]
    collapsed_tensor = torch.where(all_same_mask, temp[:, :, 0], temp[:, :, 0])
    n_masked_patches=torch.sum(collapsed_tensor != 0, dim=1, dtype=torch.long)
    n_masked_patches_tensor= torch.full((1,), fill_value=n_masked_patches.sum().item(), dtype=torch.long)
    # original ---  "n_masked_patches": torch.full((1,), fill_value=mask_indices_list.shape[0], dtype=torch.long)


    collated_masks = torch.stack(masks_list).flatten(1)
    mask_indices_list = collated_masks.flatten().nonzero()
    mask_indices_list=mask_indices_list.flatten()
    masks_weight = (1 / collated_masks.sum(-1).clamp(min=1.0)).unsqueeze(-1).expand_as(collated_masks)[collated_masks]

    return {
        "collated_global_crops": collated_global_crops.to(dtype),
        "collated_local_crops": collated_local_crops.to(dtype),
        "collated_masks": collated_masks,
        "mask_indices_list": mask_indices_list,
        "masks_weight": masks_weight,
        "upperbound": upperbound,
        "n_masked_patches_per_batch": n_masked_patches,
        "n_masked_patches_tensor": n_masked_patches_tensor,
        "new_mask_index_lst": collapsed_tensor,
    }

