import torch
import numpy as np

def dataCollect(state_dim,
                action_dim,
                memory,
                ranker,
                trainset,
                click_model,
                sample_iteration = 100,
                capacity = 1e6
            ):
    query_set = trainset.get_all_querys()
    index = np.arange(len(query_set))
    print(f"num of query: {len(query_set)}")

    query_count = 0
    for _ in range(sample_iteration):
        np.random.shuffle(index)
        for i in index:
            query_count += 1
            qid = query_set[i]
            print(f"index {query_count} qid {qid} memory status {len(memory)}/{capacity}")
            result_list = ranker.get_query_result_list(trainset, qid)
            clicked_doces, click_labels, _ = click_model.simulate(qid, result_list, trainset)
            if len(clicked_doces) == 0:
                continue

            state = np.zeros(state_dim, dtype=np.float32)
            next_state = np.zeros(state_dim, dtype=np.float32)
            chosen = np.ones(len(result_list), dtype=bool) 
            for i in range(len(result_list)):
                action = trainset.get_features_by_query_and_docid(qid, result_list[i]).astype(np.float32)
                state = next_state
                reward = 1/np.log2(i+2) if click_labels[i] == 1 else 0
                next_state[:action_dim] = action + i/(i+1)*state[:action_dim]
                if action_dim+i < state_dim:
                    next_state[action_dim+i] = reward
                done = 1 if i==len(result_list)-1 else 0

                memory.push(torch.tensor(state, dtype=torch.float32).reshape(1,-1),
                            torch.tensor(action, dtype=torch.float32).reshape(1,-1),
                            torch.tensor(next_state, dtype=torch.float32).reshape(1,-1),
                            torch.tensor([[reward]], dtype=torch.float32),
                            torch.tensor([[done]], dtype=torch.int),
                            torch.tensor(chosen),
                            qid)
                chosen[int(result_list[i])] = False        
            