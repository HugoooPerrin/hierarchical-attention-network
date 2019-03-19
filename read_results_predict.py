import sys
import json
import numpy as np

from keras.models import Model
from keras.layers import Input, Embedding, Dropout, Bidirectional, GRU, CuDNNGRU, TimeDistributed, Dense

# = = = = = = = = = = = = = = =

is_GPU = True

path_to_code = '/home/hugoperrin/Bureau/X/Cours/Advanced Learning for Text and Graph Data/Challenge/starting_kit/'
path_to_data = '/home/hugoperrin/Bureau/X/Cours/Advanced Learning for Text and Graph Data/Challenge/data/'
sys.path.insert(0, path_to_code)

# = = = = = = = = = = = = = = =

from AttentionWithContext import AttentionWithContext

def bidir_gru(my_seq,n_units,is_GPU):
    '''
    just a convenient wrapper for bidirectional RNN with GRU units
    enables CUDA acceleration on GPU
    # regardless of whether training is done on GPU, model can be loaded on CPU
    # see: https://github.com/keras-team/keras/pull/9112
    '''
    if is_GPU:
        return Bidirectional(CuDNNGRU(units=n_units,
                                      return_sequences=True),
                                      merge_mode='concat', weights=None)(my_seq)
    else:
        return Bidirectional(GRU(units=n_units,
                                 activation='tanh', 
                                 dropout=0.0,
                                 recurrent_dropout=0.0,
                                 implementation=1,
                                 return_sequences=True,
                                 reset_after=True,
                                 recurrent_activation='sigmoid'),
                             merge_mode='concat', weights=None)(my_seq)

# = = = = = = = = = = = = = = =

docs = np.load(path_to_data + 'documents.npy')
embeddings = np.load(path_to_data + 'embeddings.npy')

with open(path_to_data + 'train_idxs.txt', 'r') as file:
    train_idxs = file.read().splitlines()

with open(path_to_data + 'test_idxs.txt', 'r') as file:
    test_idxs = file.read().splitlines()

train_idxs = [int(elt) for elt in train_idxs]
test_idxs = [int(elt) for elt in test_idxs]

docs_test = docs[test_idxs,:,:]

# = = = = = TRAINING RESULTS = = = = = 

for tgt in range(4):
    
    print('* * * * * * *',tgt,'* * * * * * *')
    
    with open(path_to_data + '/model_history_' + str(tgt) + '.json', 'r') as file:
        hist = json.load(file)
    
    val_mse = hist['val_loss']
    val_mae = hist['val_mean_absolute_error']
    
    min_val_mse = min(val_mse)
    min_val_mae = min(val_mae)
    
    best_epoch = val_mse.index(min_val_mse) + 1
    
    print('best epoch:',best_epoch)
    print('best val MSE',round(min_val_mse,3))
    print('best val MAE',round(min_val_mae,3))

# = = = = = PREDICTIONS = = = = =     

all_preds_mean = []
all_preds_han = []

for tgt in range(4):
    
    print('* * * * * * *',tgt,'* * * * * * *')
    
    # * * * mean baseline * * * 
    
    with open(path_to_data + 'targets/train/target_' + str(tgt) + '.txt', 'r') as file:
        target = file.read().splitlines()
    
    target = np.array(target).astype('float')
    target_mean = np.mean(target)
    all_preds_mean.append([target_mean]*len(test_idxs))
    
    # * * * HAN * * * 
    
    # relevant hyper-parameters
    n_units = 50
    drop_rate = 0 # prediction mode
    
    sent_ints = Input(shape=(docs_test.shape[2],))
    
    sent_wv = Embedding(input_dim=embeddings.shape[0],
                        output_dim=embeddings.shape[1],
                        weights=[embeddings],
                        input_length=docs_test.shape[2],
                        trainable=False,
                        )(sent_ints)
    
    sent_wv_dr = Dropout(drop_rate)(sent_wv)
    sent_wa = bidir_gru(sent_wv_dr,n_units,is_GPU)
    sent_att_vec,word_att_coeffs = AttentionWithContext(return_coefficients=True)(sent_wa)
    sent_att_vec_dr = Dropout(drop_rate)(sent_att_vec)                      
    sent_encoder = Model(sent_ints,sent_att_vec_dr)
    
    doc_ints = Input(shape=(docs_test.shape[1],docs_test.shape[2],))
    sent_att_vecs_dr = TimeDistributed(sent_encoder)(doc_ints)
    doc_sa = bidir_gru(sent_att_vecs_dr,n_units,is_GPU)
    doc_att_vec,sent_att_coeffs = AttentionWithContext(return_coefficients=True)(doc_sa)
    doc_att_vec_dr = Dropout(drop_rate)(doc_att_vec)
    
    preds = Dense(units=1, activation='sigmoid')(doc_att_vec_dr)
    
    model = Model(doc_ints, preds)
    
    model.load_weights(path_to_data + 'model_' + str(tgt))
    
    all_preds_han.append(model.predict(docs_test).tolist())

# flatten
all_preds_mean = [elt for sublist in all_preds_mean for elt in sublist]
all_preds_han = [elt[0] for sublist in all_preds_han for elt in sublist]

# write predictions in Kaggle format
with open(path_to_data + 'predictions_mean.txt', 'w') as file:
    file.write('id,pred\n')
    for idx,pred in enumerate(all_preds_mean):
        pred = format(pred, '.7f')
        file.write(str(idx) + ',' + pred + '\n')

with open(path_to_data + 'predictions_han.txt', 'w') as file:
    file.write('id,pred\n')
    for idx,pred in enumerate(all_preds_han):
        pred = format(pred, '.7f')
        file.write(str(idx) + ',' + pred + '\n')
