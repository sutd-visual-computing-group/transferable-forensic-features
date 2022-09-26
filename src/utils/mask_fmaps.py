
# Import base libraries
import os, sys, math

# Import torch libraries
import torch

# Import other libraries
from tqdm import tqdm


def feature_map_dropout(feature_map_idx):
    """
    Mask the output produced by the specific channel
    """
    def hook(module, inputs, outputs):
        if not module.training: # (B, C, H, w)
            outputs[:, feature_map_idx, :, :] = outputs[:, feature_map_idx, :, :]*0.0 # multiply the channel output by 0.0
        else:
            raise NotImplementedError("Please set your model in evalutation mode for sensitivity assessments")
    return hook


def mask_target_channels(model, topk_dict):
    """
    Mask target channels (Can be topk or lowk channels)
        topk_dict i.e.: layer0.conv1.#33(T=64)
    
    The register_forward_hook function can modify the output directly. (It also cannot modify the inputs)
    """
    hooks = []

    with tqdm(total=len(topk_dict)) as pbar:
        for filter_name in topk_dict:
            total_filters = filter_name.split('.#')[-1].split('=')[1][:-1]
            feature_map_idx = int(filter_name.split('.#')[-1].split('(')[0])

            for i, (name, mod) in enumerate(model.named_modules()):
                # print(i)
                if filter_name.split("#")[0][:-1] == mod.auto_name:
                    #print("hit >> ", filter_name, feature_map_idx, total_filters)
                    pbar.set_description("dropout >> filter_name : {}, index : {}/{}".format(filter_name, 
                                        feature_map_idx, total_filters))
                    hook = mod.register_forward_hook(feature_map_dropout(feature_map_idx))
                    hooks.append(hook)

            pbar.update(1)

    return model, hooks



def mask_random_channels(model, topk, topk_dict, all_feature_maps):
    import random
    """
    Mask random channels not in the topk dict
        topk_dict i.e.: layer0.conv1.#33(T=64)
    
    The register_forward_hook function can modify the output directly. (It also cannot modify the inputs)
    """
    hooks = []
    all_feature_maps = set(random.choices(list(all_feature_maps), k=len(all_feature_maps))) # Shuffle
    total = 0

    if topk == 0:
        return model, hooks

    with tqdm(total=len(topk_dict)) as pbar:
        for feature_map_name in all_feature_maps:
            if feature_map_name in topk_dict:
                pbar.set_description("{} occurs in topk, skipping".format(feature_map_name))
                #print("{} occurs in topk, skipping".format(feature_map_name))
                continue
            
            total_filters = int(feature_map_name.split('.#')[-1].split('=')[1][:-1])
            feature_map_idx  = int(feature_map_name.split('.#')[-1].split('(')[0])

            # Attach masking hook
            for i, (name, mod) in enumerate(model.named_modules()):
                if feature_map_name.split("#")[0][:-1] == mod.auto_name:
                    # print("hit >> ", filter_name, channel_idx, total_filters)
                    pbar.set_description("dropout >> filter_name : {}, index : {}/{}".format(feature_map_name, 
                                        feature_map_idx, total_filters))
                    hook = mod.register_forward_hook(feature_map_dropout(feature_map_idx))
                    hooks.append(hook)
            
            total += 1
            pbar.update(1)

            if total == topk:
                break

    return model, hooks