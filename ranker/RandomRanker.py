import torch
import numpy as np

from ranker.AbstractRanker import AbstractRanker

class RandomRanker(AbstractRanker):
    def __init__(self, 
                num_features):
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def update_policy(self):
        pass

    def get_query_result_list(self, dataset, query, seed):
        np.random.seed(seed)
        docid_list = np.array(dataset.get_candidate_docids_by_query(query),dtype=np.int32)
        np.random.shuffle(docid_list)
        return docid_list
        
    def get_all_query_result_list(self, dataset):
        query_result_list = {}
        for query in dataset.get_all_querys():
            query_result_list[query] = self.get_query_result_list(dataset, query)
        
        return query_result_list