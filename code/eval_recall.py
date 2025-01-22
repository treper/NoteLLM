import tqdm
import csv
import pickle
import numpy as np
import torch
import sys
import random
from torch.nn import functional as F

def set_random_seed(random_seed=42):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

def case_study(recall_indexs, ranking, note_id_list, doc_note_id_list):
    query = []
    gt = []
    gt_ranking = []
    top = 5
    top_k = []
    for i in range(len(note_id_list)):
        if len(query)==50:
            break
        query.append(note_id_list[i])
        ranking_top = ranking[i][:top]
        ranking_doc = [doc_note_id_list[k.item()] for k in ranking_top]
        top_k.append(ranking_doc)
        gt.append(doc_note_id_list[ranking[i][recall_indexs[i]].item()])
        gt_ranking.append(recall_indexs[i])
    for i in range(len(query)):
        print(f'query doc:\n {query[i]},')
        print(f'ground truth is top {gt_ranking[i]}, ground truth doc:\n {gt[i]},')
        pred_doc_top = ',\n'.join(top_k[i])
        print(f'pred doc top5:\n {pred_doc_top}')
        print('-----------------------------------------')

def count_topK_recall_100w(query_features, doc_features, query_doc_index, note_id_list=None, doc_note_id_list=None, logits=None):
    chunk=500
    recall_indexs = []

    doc_features = doc_features.cuda()
    for i in tqdm.tqdm(range(0, query_features.shape[0], chunk)):
        tmp_query_features = query_features[i:i+chunk]
        tmp_query_features = tmp_query_features.cuda()
        logit = tmp_query_features @ doc_features.t()
        ranking = torch.argsort(logit, dim=1, descending=True)
        del logit

        idxes = []
        for j in range(len(ranking)):
            k = query_doc_index[i + j]
            idxes.append([j, k])
        idxes = torch.tensor(idxes).cuda()
        preds_index = torch.nonzero(ranking[idxes[:, 0]] == idxes[:, 1].unsqueeze(-1))[:, -1]
        recall_indexs.extend(preds_index.cpu().numpy().tolist())
        if i==0:
            case_study(recall_indexs, ranking, note_id_list, doc_note_id_list)

    print('recall_indexs:', len(recall_indexs))

    metrics = {}
    for k in [10, 20, 50, 100, 1000, 10000, 100000, 1000000]:
        metrics[f"R@{k}"] = 0
        for preds_index in recall_indexs:
            if preds_index < k:
                metrics[f"R@{k}"] += 1
        metrics[f"R@{k}"] /= len(query_doc_index)
    return metrics

def count_topK_recall():

    result_path=sys.argv[1]
    noteid_cache_path=sys.argv[2]
    rel_path=sys.argv[3]
    logits=sys.argv[4]
    seed=int(sys.argv[5])

    set_random_seed(seed)

    noteid_2id = np.load(noteid_cache_path,allow_pickle=True).item()
    id2noteid = {v:k for k,v in noteid_2id.items()}

    with open(result_path,'rb')as f:
        noteemb_id = pickle.load(f)
    
    embeddings = noteemb_id[logits]
    emb_id = noteemb_id['id']
    noteids = [id2noteid[eid] for eid in emb_id]
    note_id_dict = {note_id:embedding for note_id, embedding in zip(noteids, embeddings) if note_id in noteid_2id}

    print(len(note_id_dict))
    
    f = open(rel_path, "r")

    note_id_list = {}
    id2note={}
    gnote_id_list = {}
    id2gnote={}
    query_doc_list = []
    f_csv = csv.reader(f)

    for idx, row in tqdm.tqdm(enumerate(f_csv)):
        noteid, gnoteid = row[0], row[1]
        if noteid not in note_id_dict or gnoteid not in note_id_dict:
            continue
        if noteid not in note_id_list \
                and noteid not in gnote_id_list \
                and gnoteid not in note_id_list \
                and gnoteid not in gnote_id_list:
            query_doc_list.append([len(note_id_list), len(gnote_id_list)])
            id2note[len(note_id_list)] = noteid
            note_id_list[noteid] = len(note_id_list)
            id2gnote[len(gnote_id_list)] = gnoteid
            gnote_id_list[gnoteid] = len(gnote_id_list)

    query_features = []
    for q_id in range(len(note_id_list)):
        note_id = id2note[q_id]
        query_features.append(np.expand_dims(note_id_dict[note_id], axis=0))
    print("query_features:", len(query_features))
    query_features = np.concatenate(query_features, axis=0)
    print(query_features.shape)

    doc_note_id_list = list(set(note_id_dict.keys()) - set(note_id_list.keys()))
    id2doc = {i:v for i,v in enumerate(doc_note_id_list)}
    doc2id = {v:i for i,v in enumerate(doc_note_id_list)}
    doc_features = []
    for doc_id in range(len(id2doc)):
        note_id = id2doc[doc_id]
        doc_features.append(np.expand_dims(note_id_dict[note_id], axis=0))
    print("doc_features:", len(doc_features))
    doc_features = np.concatenate(doc_features, axis=0)
    print(doc_features.shape)

    query_doc_index = [0]*len(query_doc_list)
    for q_id, g_id in tqdm.tqdm(query_doc_list):
        g_note_id = id2gnote[g_id]
        d_id = doc2id[g_note_id]
        query_doc_index[q_id]=d_id
    print("query_doc_index:", len(query_doc_index))
        
    query_features = torch.tensor(query_features)
    doc_features = torch.tensor(doc_features)
    metrics = count_topK_recall_100w(query_features, doc_features, query_doc_index, id2note, id2doc, logits)
    print(metrics)

if __name__ == '__main__':
    count_topK_recall()
