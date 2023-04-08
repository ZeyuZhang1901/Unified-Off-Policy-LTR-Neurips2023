import numpy as np
import torch
import os

def read_performance(
        performance_file_folder,
        folder_num,
        click_type,
):
    '''read performance from file.
        performance_file_path: str, path to the folder of performance files
        folder_num: int, number of the folder
        click_type: str, 'pbm', 'cascade' or 'dcm'

    '''
    file_path = os.path.join(performance_file_folder,\
                            "fold" + str(folder_num),\
                            click_type,\
                            'minprob_0.1_eta_1_run1',\
                            'performance_test.txt')
    with open(file_path, 'r') as f:
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

if __name__ == "__main__":
    # performance_file_folder = r"run_demo\\results_rand\\SAC_CQL_LSTM_"
    # performance_file_folder = r"run_demo\\results_svm\\SAC_CQL_LSTM_"
    # performance_file_folder = r"current_best2\\results_rand\\SAC_CQL_LSTM_"
    # performance_file_folder = r"current_best2\\results_svm\\SAC_CQL_LSTM_"
    performance_file_folder = r"current_best2\\results_rand\\SAC_CQL_ATTENTION_"
    # performance_file_folder = r"current_best2\\results_svm\\SAC_CQL_ATTENTION_"
    folder_nums = [1]
    # run_nums = [26,27,28,29,30]
    run_nums = [4,5]
    click_types = ['pbm', 'dcm', 'cascade']
    for run_num in run_nums:
        for folder_num in folder_nums:
            for click_type in click_types:
                performance = read_performance(
                    performance_file_folder=performance_file_folder+str(run_num),
                    folder_num=folder_num,
                    click_type=click_type,
                )
                print("Run: %d\tFolder: %d\tclick type: %s\t\n\t%s"
                    % (
                        run_num,
                        folder_num,
                        click_type,
                        "\n\t".join(
                            ["%s:%.3f" % (key, value) for key, value in performance.items()]
                        ),
                    )
                )
                print("\n")