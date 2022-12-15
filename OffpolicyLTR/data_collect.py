import torch
import numpy as np

def dataCollect(state_dim,
                action_dim,
                memory,
                ranker,
                trainset,
                click_model,
                sample_iteration = 100,
                capacity = 1e6,
                end_pos = 10,
            ):
    query_set = trainset.get_all_querys()
    index = np.arange(len(query_set))
    print(f"num of query: {len(query_set)}")

    query_count = 0
    for _ in range(sample_iteration):
        np.random.shuffle(index)
        for ind in index:
            query_count += 1
            qid = query_set[ind]
            print(f"index {query_count} qid {qid} memory status {len(memory)}/{capacity}")
            result_list = ranker.get_query_result_list(trainset, qid)
            # print(f"result list: {result_list}")
            clicked_doces, click_labels, _ = click_model.simulate(qid, result_list, trainset)
            # print(f"click doc: {clicked_doces}")
            # print(f"click label: {click_labels}")

            state = np.zeros(state_dim, dtype=np.float32)
            next_state = np.zeros(state_dim, dtype=np.float32)
            chosen = np.ones(len(result_list), dtype=bool) 
            for j in range(len(result_list)):
                if j>=end_pos:  # only record tuples before end_pos
                    break
                # action
                action = trainset.get_features_by_query_and_docid(qid, result_list[j]).astype(np.float32)
                # state
                state = next_state
                # reward
                reward = 1/np.log2(j+2) if click_labels[j] == 1 else 0
                # next state
                # next_state[:action_dim] = action + j/(j+1)*state[:action_dim]
                next_state[action_dim+j] = reward
                next_state[action_dim+end_pos+j] = 1  # one-hot, indicate current position
                if j>0:
                    next_state[action_dim+end_pos+j-1] = 0 
                # done
                done = 1 if j==len(result_list)-1 or j==end_pos-1 else 0

                memory.push(torch.tensor(state, dtype=torch.float32).reshape(1,-1),
                            torch.tensor(action, dtype=torch.float32).reshape(1,-1),
                            torch.tensor(next_state, dtype=torch.float32).reshape(1,-1),
                            torch.tensor([[reward]], dtype=torch.float32),
                            torch.tensor([[done]], dtype=torch.int),
                            torch.tensor(chosen),
                            qid)
                chosen[int(result_list[j])] = False
            