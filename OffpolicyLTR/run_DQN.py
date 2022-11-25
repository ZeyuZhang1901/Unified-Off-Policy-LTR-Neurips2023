import sys
sys.path.append('/home/zeyuzhang/Projects/myOLTR/')
from dataset import LetorDataset
from data_collect import dataCollect
from clickModel.PBM import PBM
from clickModel.CM import CM
from ranker.DQNRanker import DQNRanker
from network.Memory import Memory
from utils import evl_tool
import multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

# %%
def run(train_set, 
        test_set,
        ranker, 
        memory,
        num_iteration,
        end_pos
    ):

    ndcg_scores = []
    q_values = []
    target_q_values = []
    losses = []
    
    for i in range(num_iteration):
        q, target_q, loss = ranker.update_policy(memory, train_set)
        q_values.append(q)
        target_q_values.append(target_q)
        losses.append(loss)
        print(f"iter {i+1}: q_value {q} target q value {target_q} loss {loss}")

        # evaluate
        if i % 100 == 0:
            all_result = ranker.get_all_query_result_list(test_set)
            ndcg = evl_tool.average_ndcg_at_k(test_set, all_result, end_pos)
            ndcg_scores.append(ndcg)
            print(f"eval {int((i+1)/100)} ndcg {ndcg}")
        
    return ndcg_scores, q_values, target_q_values, losses

# %%
def job(model_type, 
        sample_iteration,
        state_dim, 
        action_dim,
        memory,
        f, 
        train_set, 
        test_set, 
        output_fold
    ):

    if model_type == "perfect":
        pc = [0.0, 0.2, 0.4, 0.8, 1.0]
        ps = [0.0, 0.0, 0.0, 0.0, 0.0]
    elif model_type == "navigational":
        pc = [0.05, 0.3, 0.5, 0.7, 0.95]
        ps = [0.2, 0.3, 0.5, 0.7, 0.9]
    elif model_type == "informational":
        pc = [0.4, 0.6, 0.7, 0.8, 0.9]
        ps = [0.1, 0.2, 0.3, 0.4, 0.5]
    #
    # if model_type == "perfect":
    #     pc = [0.0, 0.5, 1.0]
    #     ps = [0.0, 0.0, 0.0]
    # elif model_type == "navigational":
    #     pc = [0.05, 0.5, 0.95]
    #     ps = [0.2, 0.5, 0.9]
    # elif model_type == "informational":
    #     pc = [0.4, 0.7, 0.9]
    #     ps = [0.1, 0.3, 0.5]

    # cm = PBM(pc, 1)
    cm = CM(pc, 1)
    for r in range(1, 2):
        # np.random.seed(r)
        print("DQN MQ2008 fold{} {}  run{} start!".format(f, model_type, r))
        # print("DQN MQ2007 fold{} {}  run{} start!".format(f, model_type, r))
        # print("DQN MSLR10K fold{} {}  run{} start!".format(f, model_type, r))
        ranker = DQNRanker(state_dim, action_dim, LR, BATCH_SIZE, DISCOUNT, TAU)
        dataCollect(state_dim, action_dim, memory, ranker, train_set, cm, sample_iteration, CAPACITY)
        ndcg_scores, q_values, target_q_values, losses = run(train_set, test_set, ranker, memory, NUM_INTERACTION, END_POS)
        writer = SummaryWriter("{}/fold{}/{}_run{}_ndcg/".format(output_fold, f, model_type, r))
        for i in range(len(ndcg_scores)):
            writer.add_scalar('ndcg',ndcg_scores[i], i+1)
        for j in range(len(losses)):
            writer.add_scalar('policy', q_values[j], j+1)
            writer.add_scalar('target', target_q_values[j], j+1)
            writer.add_scalar('avg_loss', losses[j], j+1)
        writer.close()

if __name__ == "__main__":

    END_POS = 10
    # FEATURE_SIZE = 136
    # ACTION_DIM = 136
    FEATURE_SIZE = 46
    ACTION_DIM = 46
    STATE_DIM = ACTION_DIM + END_POS
    BATCH_SIZE = 256
    NUM_INTERACTION = 10000
    SAMPLE_ITERATION = 1
    CAPACITY = 1e6
    DISCOUNT = 0.9
    TAU = 0.005
    LR = 1e-3
    END_POS = 10

    # click_models = ["informational", "perfect", "navigational"]
    # click_models = ["informational", "perfect"]
    click_models = ["perfect"]

    # dataset_fold = "/home/zeyuzhang/Projects/myLTR/datasets/MSLR10K"
    # output_fold = "results/MSLR10K/DQN"
    # dataset_fold = "/home/zeyuzhang/Projects/myLTR/datasets/2007_mq_dataset"
    # output_fold = "results/MQ2007/DQN"
    dataset_fold = "/home/zeyuzhang/Projects/myLTR/datasets/2008_mq_dataset"
    output_fold = "results/MQ2008/DQN"

    # for 5 folds
    for f in range(1, 2):
        training_path = "{}/Fold{}/train.txt".format(dataset_fold, f)
        test_path = "{}/Fold{}/test.txt".format(dataset_fold, f)
        # train_set = LetorDataset(training_path, FEATURE_SIZE, query_level_norm=True)
        # test_set = LetorDataset(test_path, FEATURE_SIZE, query_level_norm=True)
        train_set = LetorDataset(training_path, FEATURE_SIZE, query_level_norm=False)
        test_set = LetorDataset(test_path, FEATURE_SIZE, query_level_norm=False)
        memory = Memory(capacity=int(CAPACITY))

        processors = []
        # for 3 click_models
        for click_model in click_models:
            p = mp.Process(target=job, args=(click_model, SAMPLE_ITERATION, STATE_DIM, ACTION_DIM, memory, f, 
                    train_set, test_set, output_fold))
            p.start()
            processors.append(p)
    for p in processors:
        p.join()