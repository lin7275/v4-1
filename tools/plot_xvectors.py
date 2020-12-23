#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 16:13:11 2020

@author: mwmak
"""

import h5py
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Load x-vectors from .h5 file
def load_xvectors_h5(h5file, n_samples=1000, start=0):
    print('Loading %s' % h5file)
    with h5py.File(h5file, 'r') as f:
        X = f['X'][start:start+n_samples, :]
        n_frames = f['n_frames'][start:start+n_samples]
        spk_ids = f['spk_ids'][start:start+n_samples]
        spk_path = f['spk_path'][start:start+n_samples]
    return X, n_frames, spk_ids, spk_path

if __name__ == '__main__':
    start = 20000
    X1, _, _, _ = load_xvectors_h5('eval_on_v19/h5/xvector/vox2.h5', start=start)
    X2, _, _, _ = load_xvectors_h5('eval_on_v19/h5/densenet/vox2.h5', start=start)
    
    print('Performing TSNE')
    X1_prj = TSNE(random_state=20150101).fit_transform(X1)
    X2_prj = TSNE(random_state=20150101).fit_transform(X2)

    print('Plot t-SNE vectors')
    _, ax = plt.subplots()
    ax.scatter(X1_prj[:,0], X1_prj[:,1], marker='.', label='X-vector')
    ax.scatter(X2_prj[:,0], X2_prj[:,1], marker='.', label='DenseNet')
    ax.legend()
    plt.show()
    
    