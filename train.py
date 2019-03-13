#!/usr/bin/env python
# -*- coding: utf-8 -*-



"""
"""



#=========================================================================================================
#================================ 0. MODULE



import os 

import numpy as np

from torch import nn
import torch.optim
import torch.backends.cudnn as cudnn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import mean_squared_error

from models import HAN
from utils import predict

from datetime import datetime
from dateutil.relativedelta import relativedelta

def diff(t_a, t_b):
    t_diff = relativedelta(t_a, t_b)
    return '{h}h {m}m {s}s'.format(h=t_diff.hours, m=t_diff.minutes, s=t_diff.seconds)


DEVICE = 'cuda:0'
cudnn.benchmark = True


#=========================================================================================================
#================================ 1. DATA



BATCH_SIZE = 64
TARGET_IDX = 0


path_to_data = '../data/'

# Loading data
print('Loading data', end='...')
docs = np.load(path_to_data + 'documents.npy')
embeddings = np.load(path_to_data + 'embeddings.npy')

with open(path_to_data + 'train_idxs.txt', 'r') as file:
    train_idxs = file.read().splitlines()
    
train_idxs = [int(elt) for elt in train_idxs]

# Train valid split
idxs_select_train = np.random.choice(range(len(train_idxs)), size=int(len(train_idxs) * 0.80), replace=False)
idxs_select_val = np.setdiff1d(range(len(train_idxs)),idxs_select_train)

train_idxs_new = [train_idxs[elt] for elt in idxs_select_train]
val_idxs = [train_idxs[elt] for elt in idxs_select_val]

docs_train = torch.LongTensor(docs[train_idxs_new,:,:])
docs_val = torch.LongTensor(docs[val_idxs,:,:])

# Loading targets
target_train = []
target_val = []

with open(path_to_data + 'targets/train/target_' + str(TARGET_IDX) + '.txt', 'r') as file:
    target = file.read().splitlines()

target_train = torch.FloatTensor(np.array([target[elt] for elt in idxs_select_train]).astype('float'))
target_val = np.array([target[elt] for elt in idxs_select_val]).astype('float')

# To torch loader
train_dataset = TensorDataset(docs_train, target_train)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)

valid_dataset = TensorDataset(docs_val)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)

# loading embedding
embeddings = np.load(path_to_data + 'embeddings.npy')
embeddings = torch.FloatTensor(embeddings)
vocab_size = embeddings.size(0)
embedding_dim = embeddings.size(1)

print('done')



#=========================================================================================================
#================================ 2. MODELS & PARAMETERS


# Hyperparameters
LEARNING_RATE = 1e-3
DISPLAY_STEP = 100

ATTENTION_DIM = [128, 128]
BI_GRU_DIM = [128, 128]
embedding_dim = 13
DROPOUT = 0.3


# Model
net = HAN(attention_dim=ATTENTION_DIM, bi_gru_dim=BI_GRU_DIM, embedding_dim=embedding_dim, 
          vocab_size=vocab_size, dropout=DROPOUT)
net.load_pretrained_embeddings(embeddings, fine_tune=False)
net = net.to(DEVICE)

# Optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)

# Loss
criterion = nn.MSELoss().to(DEVICE)

# Parameters check
model_parameters = filter(lambda p: p.requires_grad, net.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print('\n>> Learning {} parameters\n'.format(params))



#=========================================================================================================
#================================ 2. MODELS & PARAMETERS




N_EPOCHS = 20


for epoch in range(N_EPOCHS):
    epoch_loss = 0.

    time = datetime.now()

    for i, (documents, target) in enumerate(train_loader):
        net.train()

        # Batch data
        documents = documents.to(DEVICE)
        target = target.to(DEVICE)

        # Forward pass
        output = net(documents)

        # Compute loss
        loss = criterion(output, target)

        # Delete previous gradients
        optimizer.zero_grad()

        # Backpropagation
        loss.backward()

        # Take a optimizer step
        optimizer.step()
        
        # Monitoring performance
        epoch_loss += loss.data.item()

        if i % DISPLAY_STEP == DISPLAY_STEP-1:

            valid_pred = predict(net, valid_loader)
            valid_loss = mean_squared_error(target_val, valid_pred)

            print('epoch %2d, training loss: %.3f, validation loss: %.3f' % (epoch, epoch_loss / i, valid_loss))

    torch.save(net.state_dict(), "../models/han_epoch{}_target{}.model".format(epoch, TARGET_IDX))
    print('\nEpoch time: ', diff(datetime.now(), time), '\n')
