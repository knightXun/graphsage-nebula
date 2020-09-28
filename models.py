import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable

import numpy as np
import time
import random
from sklearn.metrics import f1_score
from collections import defaultdict

from encoders import Encoder
from aggregators import MeanAggregator

"""
Simple supervised GraphSAGE model as well as examples running the model
on the Cora and Pubmed datasets.
"""

class SupervisedGraphSage(nn.Module):

    def __init__(self, num_classes, enc):
        super(SupervisedGraphSage, self).__init__()
        self.enc = enc
        self.xent = nn.CrossEntropyLoss()

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, enc.embed_dim))
        init.xavier_uniform(self.weight)

    def forward(self, nodes):
        embeds = self.enc(nodes)
        scores = self.weight.mm(embeds)
        return scores.t()

    def loss(self, nodes, labels):
        scores = self.forward(nodes)
        return self.xent(scores, labels.squeeze())

import numpy as np
from collections import defaultdict
from graph import ttypes
from nebula.ConnectionPool import ConnectionPool
from nebula.Client import GraphClient
from nebula.Common import *
def read_feats():
    g_ip = '192.168.8.188'
    g_port = 3699
    connection_pool = ConnectionPool(g_ip, g_port)
    client = GraphClient(connection_pool)
    auth_resp = client.authenticate('root', '')
    if auth_resp.error_code:
        raise AuthException("Auth failed")
    client.execute_query("use cora")
    query_resp = client.execute_query("LOOKUP ON paper WHERE paper.num > 0")
    num_nodes=len(query_resp.rows)
    labels = np.empty((num_nodes,1), dtype=np.int64)
    label_map = {}
    
    query_resp = client.execute_query("fetch prop ON paper 1")
    feature_str = query_resp.rows[0].columns[2].get_str().decode().replace(' ', '')
    feat_data = np.zeros((num_nodes, len(feature_str)))
    
    adj_lists = defaultdict(set)
    
    for i in range(num_nodes):
        query_resp = client.execute_query("fetch prop ON paper {}".format(i))
        feature_str = query_resp.rows[0].columns[2].get_str().decode()
        label = query_resp.rows[0].columns[3].get_str().decode()
        feature = np.fromstring(str(feature_str), dtype=int,sep=' ')
        feat_data[i] = feature
        if not label in label_map:
            label_map[label] = len(label_map)
        labels[i] = label_map[label]
        
        edge_query = client.execute_query("GO FROM {} OVER cite".format(i+1))
        if edge_query.rows is None:
            continue
        for row in edge_query.rows:    
            paperId = row.columns[0].get_id()
            adj_lists[i].add(int(paperId))
            adj_lists[int(paperId)].add(i)
            
    return feat_data, labels, adj_lists

def run_cora():
    np.random.seed(1)
    random.seed(1)
    num_nodes = 2708
    feat_data, labels, adj_lists = read_feats()
    features = nn.Embedding(2708, 1433)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
   # features.cuda()

    agg1 = MeanAggregator(features, cuda=True)
    enc1 = Encoder(features, 1433, 128, adj_lists, agg1, gcn=True, cuda=False)
    agg2 = MeanAggregator(lambda nodes : enc1(nodes).t(), cuda=False)
    enc2 = Encoder(lambda nodes : enc1(nodes).t(), enc1.embed_dim, 128, adj_lists, agg2,
            base_model=enc1, gcn=True, cuda=False)
    enc1.num_samples = 5
    enc2.num_samples = 5

    graphsage = SupervisedGraphSage(7, enc2)
#    graphsage.cuda()
    rand_indices = np.random.permutation(num_nodes)
    test = rand_indices[:1000]
    val = rand_indices[1000:1500]
    train = list(rand_indices[1500:])

    optimizer = torch.optim.SGD(filter(lambda p : p.requires_grad, graphsage.parameters()), lr=0.7)
    times = []
    for batch in range(100):
        batch_nodes = train[:256]
        random.shuffle(train)
        start_time = time.time()
        optimizer.zero_grad()
        loss = graphsage.loss(batch_nodes, 
                Variable(torch.LongTensor(labels[np.array(batch_nodes)])))
        loss.backward()
        optimizer.step()
        end_time = time.time()
        times.append(end_time-start_time)
        print(batch, loss.item())

    val_output = graphsage.forward(val) 
    print("Validation F1:", f1_score(labels[val], val_output.data.numpy().argmax(axis=1), average="micro"))
    print("Average batch time:", np.mean(times))

if __name__ == "__main__":
    run_cora()
