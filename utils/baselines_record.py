import sys
import os

'''This python script is used to record all metrics of baselines and our algorithm in one text file.
'''

def read_one_performance(result_file_path):
    '''read one performance from one file.
        result_file_path: str, path to the performance file
    '''

    with open(result_file_path, 'r') as f:
        lines = f.readlines()
        performance = {}
        for line in lines:
            line = line.strip()
            if line.startswith('err_3'):
                performance['err_3'] = float(line.split(' ')[-1])
            if line.startswith('err_5'):
                performance['err_5'] = float(line.split(' ')[-1])
            if line.startswith('err_10'):
                performance['err_10'] = float(line.split(' ')[-1])
            if line.startswith('ndcg_3'):
                performance['ndcg_3'] = float(line.split(' ')[-1])
            if line.startswith('ndcg_5'):
                performance['ndcg_5'] = float(line.split(' ')[-1])
            if line.startswith('ndcg_10'):
                performance['ndcg_10'] = float(line.split(' ')[-1])
    return performance

def read_all_performances(root_file_path, algs, run_times, fold_nums, click_types):
    """read all performances from files in the root folder.

    root_file_path: str, path to the root folder of performance files
    alphas: list of str, alpha values of the experiment, e.g. ['0e0', '1e-1', '1e-2', '1e-3', '1e-4']
    run_times: list of int, run times of the experiment, e.g. [1,2,3,4,5] for 5 runs
    fold_nums: list of int, fold numbers of the experiment, e.g. [1,2,3,4,5] for 5 folds
    click_types: list of str, click types of the experiment, e.g. ['pbm', 'dcm', 'cascade']

    """

    performances = {}  # restore all performance dicts with different click types
    for click_type in click_types:
        performances[click_type] = {
            "err_3": [],
            "err_5": [],
            "err_10": [],
            "ndcg_3": [],
            "ndcg_5": [],
            "ndcg_10": [],
        }  # for each metric, restore a list of performances with different alphas
        for alpha in alphas:
            tmp_dict = {
                "err_3": [],
                "err_5": [],
                "err_10": [],
                "ndcg_3": [],
                "ndcg_5": [],
                "ndcg_10": [],
            }  # used to restore performances with different run times and fold numbers
            for fold_num in fold_nums:
                for run_time in run_times:
                    result_file_path = os.path.join(
                        root_file_path,
                        "alpha_" + alpha,
                        "fold" + str(fold_num),
                        click_type,
                        "minprob_0.1_eta_1_run" + str(run_time),
                        "performance_test.txt",
                    )
                    performance = read_one_performance(result_file_path)  # dict
                    tmp_dict["err_3"].append(performance["err_3"])
                    tmp_dict["err_5"].append(performance["err_5"])
                    tmp_dict["err_10"].append(performance["err_10"])
                    tmp_dict["ndcg_3"].append(performance["ndcg_3"])
                    tmp_dict["ndcg_5"].append(performance["ndcg_5"])
                    tmp_dict["ndcg_10"].append(performance["ndcg_10"])
            performances[click_type]["err_3"].append(
                sum(tmp_dict["err_3"]) / (len(run_times) * len(fold_nums))
            )
            performances[click_type]["err_5"].append(
                sum(tmp_dict["err_5"]) / (len(run_times) * len(fold_nums))
            )
            performances[click_type]["err_10"].append(
                sum(tmp_dict["err_10"]) / (len(run_times) * len(fold_nums))
            )
            performances[click_type]["ndcg_3"].append(
                sum(tmp_dict["ndcg_3"]) / (len(run_times) * len(fold_nums))
            )
            performances[click_type]["ndcg_5"].append(
                sum(tmp_dict["ndcg_5"]) / (len(run_times) * len(fold_nums))
            )
            performances[click_type]["ndcg_10"].append(
                sum(tmp_dict["ndcg_10"]) / (len(run_times) * len(fold_nums))
            )

    return performances