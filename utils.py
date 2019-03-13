#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
Function for torch processes
"""


#=========================================================================================================
#================================ 0. MODULE


import numpy as np


#=========================================================================================================
#================================ 1. NEURAL NETWORKS UTILS



def predict(net, loader, device='cuda:0'):
    net.eval()

    first = True

    # Get the inputs by batch to optimize GPU memory use
    for data in loader:

        data = data[0].to(device)

        batch_pred = net(data)

        del data

    # Get predicted values
        prediction = batch_pred.data.cpu().numpy()

    # Concatenate with previous batches 
        if first:
            full_prediction = prediction
            first = False
        else:
            full_prediction = np.concatenate((full_prediction, prediction), axis=0)

    return full_prediction