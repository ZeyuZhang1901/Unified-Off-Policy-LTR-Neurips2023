import os
import sys

sys.path.append(r'./')

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
    performance_file_folder = r"./results_rand/"
    # performance_file_folder = r"current_best2\\results_svm\\"
    output_file_path = r"./results_rand/metric.txt"

    folder_nums = [1]
    run_nums = [[0,1,2,3],[4,5,6,7,8,9,10,11]]
    # embed_types = ['SAC_CQL_LSTM_', 'SAC_CQL_ATTENTION_']
    # exp_types = [embed_types[0] + str(x) for x in run_nums[0]] + [embed_types[1] + str(x) for x in run_nums[1]]
    exp_types = ["SAC_CQL_ATTENTION", "SAC_CQL_LSTM"]
    click_types = ['pbm', 'dcm', 'cascade']

    fout = open(output_file_path, 'w')
    fout.write('exp_type\t\t\tfolder_num\tclick_type\terr_3 \terr_5 \terr_10\tndcg_3\tndcg_5\tndcg_10\n')
    for exp_type in exp_types:
        for folder_num in folder_nums:
            for click_type in click_types:
                performance = read_performance(
                    performance_file_folder=performance_file_folder+str(exp_type),
                    folder_num=folder_num,
                    click_type=click_type,
                )
                line = ("%-20s\t%-10d\t%-10s\t%s"
                    % (
                        exp_type,
                        folder_num,
                        click_type,
                        "\t".join(
                            ["%-10.3f" % value for value in performance.values()]
                        ),
                    )
                ) + "\n"
                fout.write(line)
    fout.close()
                