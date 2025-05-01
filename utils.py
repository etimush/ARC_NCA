import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import re
import os
import json



def get_batch(pool, x_prime, batch_size, noise_level = 0.0):
    idxs = np.random.randint(0, pool.shape[0], batch_size)
    batch = pool[idxs, :, :, :]
    
    if noise_level > 0:
        chn = pool.shape[1]
        m1 = torch.rand_like(batch) < noise_level
        #batch[:,:chn//2,...] = (batch[:,:chn//2,...]* (~m1[:,:chn//2,...]).float()) + torch.randn_like(batch[:,:chn//2,...]) * m1[:,:chn//2,...].float()
        batch[:, chn // 2:, ...] = (batch[:, chn // 2:, ...] * (~m1[:,chn//2:,...]).float()) + torch.randn_like(
            batch[:, chn // 2:, ...]) * m1[:,chn//2:,...].float()

    batch[0:1, :, :, :] = x_prime
    
    return batch, idxs


def update_problem_pool(pools, results, idxs, pool_id):
    pool_new = []
    for p in range(len(pools)):
        if p != pool_id:
            pool_new.append(pools[p])
        else:
            pool = pools[p]
            pool[idxs] = results
            pool_new.append(pool)
    return pool_new


def make_path(path):

    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Path: {path} created")
    else:
        print(f"Path: {path} already exists, all OK!")




def create_empty_json(filepath):

    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump({}, f) # Writes the empty dictionary as JSON