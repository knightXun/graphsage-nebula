from graph import ttypes
from nebula.ConnectionPool import ConnectionPool
from nebula.Client import GraphClient
from nebula.Common import *

g_ip = '192.168.8.188'
g_port = 3699
connection_pool = ConnectionPool(g_ip, g_port)
client = GraphClient(connection_pool)
auth_resp = client.authenticate('root', '')
if auth_resp.error_code:
    raise AuthException("Auth failed")

import numpy as np
from collections import defaultdict

def read_feats(client):
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

def import_cora_into_nebula():
    client.execute_query('create space if not exists cora')
    client.execute_query("use cora")
    client.execute_query("CREATE TAG if not exists paper(num int,word_attributes string,class_label string)")
    client.execute_query("CREATE EDGE if not exists cite(paper1 int, paper2 int)")
    client.execute_query("CREATE TAG INDEX if not exists paper_index ON paper(num);")

    node_map = {}
    with open("cora/cora.content") as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            client.execute_query("INSERT VERTEX paper (num, word_attributes, class_label) VALUES {}:({}, '{}', '{}')"
                                 .format(i,info[0], ' '.join(info[1:-1]), info[-1]))
            node_map[info[0]] = i

    with open("cora/cora.cites") as fp:
        for i,line in enumerate(fp):
            info = line.strip().split('\t')
            paper1 = node_map[info[0]]
            paper2 = node_map[info[1]]
            client.execute_query("INSERT EDGE cite(paper1, paper2) VALUES {}->{}:({}, {})".format(paper1, paper2, info[0], info[1]))
            client.execute_query("INSERT EDGE cite(paper1, paper2) VALUES {}->{}:({}, {})".format(paper2, paper1, info[1], info[0]))


