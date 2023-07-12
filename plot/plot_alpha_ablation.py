import sys
import os
import matplotlib.pyplot as plt

"""This python script is used to plot the alpha ablation study results.

Usage:
    python alpha_ablation_plot.py [result_file_path] [output_file_path]
"""


def read_one_performance(result_file_path):
    """read performance from one file.

    result_file_path: str, path to the file of performance results

    """

    with open(result_file_path, "r") as f:
        lines = f.readlines()
        performance = {}
        for line in lines:
            line = line.strip()
            if line.startswith("err_3"):
                performance["err_3"] = float(line.split(" ")[-1])
            if line.startswith("err_5"):
                performance["err_5"] = float(line.split(" ")[-1])
            if line.startswith("err_10"):
                performance["err_10"] = float(line.split(" ")[-1])
            if line.startswith("ndcg_3"):
                performance["ndcg_3"] = float(line.split(" ")[-1])
            if line.startswith("ndcg_5"):
                performance["ndcg_5"] = float(line.split(" ")[-1])
            if line.startswith("ndcg_10"):
                performance["ndcg_10"] = float(line.split(" ")[-1])
    return performance


def read_all_performances(root_file_path, alphas, run_times, fold_nums, click_types):
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


def plot_figures(performances, alphas, output_file_path):
    """plot performance curves of all metrics for each click type

    performances: dict, performances of all click types
    alphas: list of str, alpha values of the experiment, e.g. ['0e0', '1e-1', '1e-2', '1e-3', '1e-4']
    output_path: str, path to the output file
    """

    click_types = list(performances.keys())
    metrics = performances[click_types[0]].keys()
    alphas = [float(x) for x in alphas]

    colors = ["r", "g", "b", "y", "m", "c", "k"]

    ## plot multiple figures in one big figure (len(metrics) rows, len(click_types) columns)
    fig, axs = plt.subplots(
        len(metrics), len(click_types), figsize=(20, 30), sharex="all", sharey="row"
    )
    for click_type in click_types:
        for i, metric in enumerate(metrics):
            axs[i, click_types.index(click_type)].plot(
                alphas,
                performances[click_type][metric],
                label=metric,
                marker="^",
                color=colors[i],
            )
            if i == 0:
                axs[i, click_types.index(click_type)].set_title(click_type)
            # axs[i, click_types.index(click_type)].legend()
    fig.tight_layout()
    lines, labels = [], []
    for ax in axs[:, 0]:
        axLine, axLabel = ax.get_legend_handles_labels()
        lines.extend(axLine)
        labels.extend(axLabel)
    # legend at the bottom of the big figure
    fig.legend(lines, labels, loc="outside center right", ncol=6)
    plt.savefig(output_file_path)


if __name__ == "__main__":
    result_file_path = sys.argv[1]
    output_file_path = sys.argv[2]
    alphas = ["0e0", "1e-3", "5e-3", "1e-2", "5e-2", "1e-1", "5e-1", "1e0", "5e0", "1e1", "5e1"]
    # run_times = [1, 2, 3, 4, 5]
    run_times = [1, 2, 3]
    # run_times = [1]
    # fold_nums = [1, 2, 3, 4, 5]
    fold_nums = [1]
    click_types = ["pbm", "cascade", "dcm", "ccm"]

    performances = read_all_performances(
        root_file_path=result_file_path,
        alphas=alphas,
        run_times=run_times,
        fold_nums=fold_nums,
        click_types=click_types,
    )

    plot_figures(performances, alphas, output_file_path)
