import os
import re
import random
import numpy as np
import networkx as nx
from time import time

# = = = = = = = = = = = = = = = 

# 'atoi' and 'natural_keys' taken from: https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split('(\d+)', text)]

def random_walk(graph, node, walk_length):
    walk = [node]
    for i in range(walk_length):
        neighbors = graph.neighbors(walk[i])
        walk.append(random.choice(list(neighbors)))
    return walk

def generate_walks(graph, num_walks, walk_length):
    '''
    samples num_walks walks of length walk_length+1 from each node of graph
    '''
    graph_nodes = graph.nodes()
    n_nodes = len(graph_nodes)
    walks = []
    for i in range(num_walks):
        nodes = np.random.permutation(graph_nodes)
        for j in range(n_nodes):
            walk = random_walk(graph, nodes[j], walk_length)
            walks.append(walk)
    return walks

# = = = = = = = = = = = = = = =

pad_vec_idx = 1685894 # 0-based index of the last row of the embedding matrix (for zero-padding)

# parameters
num_walks = 5
walk_length = 10
max_doc_size = 70 # maximum number of 'sentences' (walks) in each pseudo-document

path_to_data = '/home/hugoperrin/Bureau/X/Cours/Advanced Learning for Text and Graph Data/Challenge/data/'

# = = = = = = = = = = = = = = =

def main():

    start_time = time() 

    edgelists = os.listdir(path_to_data + 'edge_lists/')
    edgelists.sort(key=natural_keys) # important to maintain alignment with the targets!

    docs = []
    for idx,edgelist in enumerate(edgelists):
        g = nx.read_edgelist(path_to_data + 'edge_lists/' + edgelist) # construct graph from edgelist
        doc = generate_walks(g,num_walks,walk_length) # create the pseudo-document representation of the graph
        docs.append(doc)
        
        if idx % round(len(edgelists)/10) == 0:
            print(idx)

    print('documents generated')
    
    # truncation-padding at the document level, i.e., adding or removing entire 'sentences'
    docs = [d+[[pad_vec_idx]*(walk_length+1)]*(max_doc_size-len(d)) if len(d)<max_doc_size else d[:max_doc_size] for d in docs] 

    docs = np.array(docs).astype('int')
    print('document array shape:',docs.shape)

    np.save(path_to_data + 'documents.npy', docs, allow_pickle=False)

    print('documents saved')
    print('everything done in %.2f seconds' % (time() - start_time))

# = = = = = = = = = = = = = = =

if __name__ == '__main__':
    main()
