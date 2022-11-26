import numpy as np
from scipy import stats

def online_mrr_at_k(clicks, k):
    reciprocal_rank = 0.0
    n_docs = len(clicks)
    for i in range(min(k, n_docs)):
        if clicks[i] > 0:
            reciprocal_rank = 1.0 / (1.0 + i)
            break
    return reciprocal_rank

def query_ndcg_at_k(dataset, result_list, query, k):
    # try:
    #     pos_docid_set = set(dataset.get_relevance_docids_by_query(query))
    # except:
    #     return 0.0
    if len(dataset.get_relevance_docids_by_query(query)) == 0:
        return 0.0
    else:
        pos_docid_set = set(dataset.get_relevance_docids_by_query(query))

    dcg = 0.0
    for i in range(0, min(k, len(result_list))):
        docid = result_list[i]
        relevance = dataset.get_relevance_label_by_query_and_docid(query, docid)
        dcg += ((2 ** relevance - 1) / np.log2(i + 2))
    rel_set = []

    for docid in pos_docid_set:
        rel_set.append(dataset.get_relevance_label_by_query_and_docid(query, docid))
    rel_set = sorted(rel_set, reverse=True)
    n = len(pos_docid_set) if len(pos_docid_set) < k else k
    idcg = 0
    for i in range(n):
        idcg += ((2 ** rel_set[i] - 1) / np.log2(i + 2))

    ndcg = (dcg / idcg)
    return ndcg

def average_ndcg_at_k(dataset, query_result_list, k, count_bad_query=False):
    ndcg = 0.0
    num_query = 0
    for query in dataset.get_all_querys():
        # try:
        #     pos_docid_set = set(dataset.get_relevance_docids_by_query(query))
        # except:
        #     print("Query:", query, "has no relevant document!")
        #     continue
        if len(dataset.get_relevance_docids_by_query(query)) == 0:
            if count_bad_query:
                num_query += 1
            continue
        else:
            pos_docid_set = set(dataset.get_relevance_docids_by_query(query))
        dcg = 0.0
        for i in range(0, min(k, len(query_result_list[query]))):
            docid = query_result_list[query][i]
            relevance = dataset.get_relevance_label_by_query_and_docid(query, docid)
            dcg += ((2 ** relevance - 1) / np.log2(i + 2))

        rel_set = []
        for docid in pos_docid_set:
            rel_set.append(dataset.get_relevance_label_by_query_and_docid(query, docid))
        rel_set = sorted(rel_set, reverse=True)
        n = len(pos_docid_set) if len(pos_docid_set) < k else k

        idcg = 0
        for i in range(n):
            idcg += ((2 ** rel_set[i] - 1) / np.log2(i + 2))

        ndcg += (dcg / idcg)
        num_query += 1
    return ndcg / float(num_query)

def get_all_query_ndcg(dataset, query_result_list, k):
    query_ndcg = {}
    for query in dataset.get_all_querys():
        try:
            pos_docid_set = set(dataset.get_relevance_docids_by_query(query))
        except:
            # print("Query:", query, "has no relevant document!")
            query_ndcg[query] = 0
            continue
        dcg = 0.0
        for i in range(0, min(k, len(query_result_list[query]))):
            docid = query_result_list[query][i]
            relevance = dataset.get_relevance_label_by_query_and_docid(query, docid)
            dcg += ((2 ** relevance - 1) / np.log2(i + 2))

        rel_set = []
        for docid in pos_docid_set:
            rel_set.append(dataset.get_relevance_label_by_query_and_docid(query, docid))
        rel_set = sorted(rel_set, reverse=True)
        n = len(pos_docid_set) if len(pos_docid_set) < k else k

        idcg = 0
        for i in range(n):
            idcg += ((2 ** rel_set[i] - 1) / np.log2(i + 2))

        ndcg = (dcg / idcg)
        query_ndcg[query] = ndcg
    return query_ndcg

def get_query_rel_at_k(dataset, query, result_list, k):
    rel_list = []
    n = len(result_list) if len(result_list) < k else k
    for i in range(n):
        rel_list.append(dataset.get_relevance_label_by_query_and_docid(query, result_list[i]))
    return rel_list

def get_all_query_rel(dataset, query_result_list, k):
    query_rel_list = {}
    for query in dataset.get_all_querys():
        query_rel_list[query] = get_query_rel_at_k(dataset, query, query_result_list[query], k)
    return query_rel_list

def get_ideal_rel_at_k(dataset, k):
    ideal_rel_list = {}
    for query in dataset.get_all_querys():
        doc_list = dataset.get_candidate_docids_by_query(query)
        rel_list = get_query_rel_at_k(dataset, query, doc_list, len(doc_list))
        ideal_rel_list[query] =  sorted(rel_list, reverse=True)[:k]
    return ideal_rel_list

def write_performance(path, dataset, ranker, end_pos):
    with open(path, 'a') as fout :
            query_result_list = ranker.get_all_query_result_list(dataset)
            all_rel_lists = get_all_query_rel(dataset, query_result_list, end_pos)
            all_ideal_rel_lists = get_ideal_rel_at_k(dataset, end_pos)
            for query in dataset.get_all_querys():
                line = f"qid {query}:\nrank list:\t{all_rel_lists[query]}\nideal list:\t{all_ideal_rel_lists[query]}\n"
                fout.write(line)

def ttest(l1, l2):
    _, p = stats.ttest_ind(l1, l2, equal_var=False)
    return p
