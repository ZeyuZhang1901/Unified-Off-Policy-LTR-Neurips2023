import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from scipy.stats import sem, t
from tensorboard.backend.event_processing import event_accumulator

sys.path.append("/home/zeyuzhang/LearningtoRank/codebase/myLTR/")
whole_path = "/home/zeyuzhang/LearningtoRank/codebase/myLTR/"

COLORS = [
    "blue",
    "green",
    "red",
    "cyan",
    "magenta",
    "yellow",
    "black",
    "brown",
    "purple",
    "pink",
    "olive",
    "gray",
    "skyblue",
    "gold",
]
LINE_STYLES = ["-", "--", "-."]

## to show color tables


def smooth(data, weight=0.9):
    """
    Smooth data with filter

    Args:
        - `data`: list of values
        - `weight`: smooth weight
    """
    smoothed = []
    last = data[0]
    for point in data:
        smooth_val = weight * last + (1 - weight) * point
        smoothed.append(smooth_val)
        last = smooth_val
    return np.array(smoothed)


def plot_one_curve(
    path,
    alg_name,
    click_type,
    mode_type,
    metric_type,
    folds,
    runs,
    num_iterations,
    color,
    line_style,
    smooth_weight,
):
    """
    Plot one curve under certain dataset, algorithm, click type and model type.

    Args:
        - `path`: str, the path of the data, under certain dataset (e.g. MQ2008/MSLRWEB10K)
        - `alg_name`: str, name of the alg (e.g. DQN, DoubleDQN)
        - `click_type`: str, name of the click model (e.g. pbm, cascade)
        - `mode_type`: str, name of the user model (e.g. perfect, informational, navigational)
        - `metric_type`: str, name of the metric (e.g. Loss_Loss)
        - `folds`: list of integers, indicate the number of the folders
        - `runs`: list of integers, indicate the number of the running times (test variance)
        - `num_iterations`: int, number of training iterations
        - `color`: int, index of color
        - `line_style`: int, index of line style
        - `smooth_weight`: float, degree of smooth
    """

    result = None
    for f in folds:
        for r in runs:
            current_path = f"{path}/{alg_name}/fold{f}/{click_type}_{mode_type}_run{r}/{metric_type}"
            filename = current_path + "/" + os.listdir(current_path)[0]
            ea = event_accumulator.EventAccumulator(filename)
            ea.Reload()
            keys = ea.scalars.Keys()
            values = [i.value for i in ea.scalars.Items(keys[0])]
            values = smooth(values, weight=smooth_weight)
            if result is None:
                result = np.zeros(values.shape[0])
            result = np.vstack((result, values))
    result = result[1:].T
    result_mean = np.mean(result, axis=1)
    result_std_err = sem(result, axis=1)
    result_h = result_std_err * t.ppf((1 + 0.95) / 2, df=25 - 1)  # df refer to n-1
    result_low = np.subtract(result_mean, result_h)
    result_high = np.add(result_mean, result_h)

    (curve,) = plt.plot(
        np.linspace(0, num_iterations, num=result.shape[0]),
        result_mean,
        color=COLORS[color],
        alpha=1,
        linestyle=LINE_STYLES[line_style],
    )
    # plt.fill_between(
    #     np.linspace(0, num_iterations, num=result.shape[0]),
    #     result_low,
    #     result_high,
    #     color=COLORS[color],
    #     alpha=0.2,
    # )
    return curve


def plot_with_fixed_click_model(
    data_path,
    save_path,
    datasets,
    alg_types,
    model_types,
    click_types,
):
    print("Each plot with fixed click model.")

    metrics = [
        "Validation_mrr_3",
        "Validation_mrr_5",
        "Validation_mrr_10",
        "Validation_ndcg_3",
        "Validation_ndcg_5",
        "Validation_ndcg_10",
    ]
    folds = list(range(1, 2))
    # runs = list(range(1, 4))
    runs = list(range(1, 2))

    num_iterations = 10000
    # num_iterations = 50000
    # smooth_weight = 0.9
    smooth_weight = 0.1

    for dataset in datasets:  # for each dataset
        if not os.path.exists(save_path + f"fix_click_model/{dataset}/"):
            os.makedirs(save_path + f"fix_click_model/{dataset}/")
        for metric in metrics:  # for each metric
            color_index = 0
            linestyle_index = 0  # indicate different model_type
            for click_type in click_types:
                labels, curves = [], []
                plt.title(f"{metric} ({dataset}, {click_type})")
                for mode_type in model_types:
                    for alg in alg_types:  # for each alg
                        labels.append(f"{alg}_{mode_type}")
                        curves.append(
                            plot_one_curve(
                                path=data_path + f"{dataset}",
                                alg_name=alg,
                                click_type=click_type,
                                mode_type=mode_type,
                                metric_type=metric,
                                folds=folds,
                                runs=runs,
                                num_iterations=num_iterations,
                                color=color_index,
                                line_style=linestyle_index,
                                smooth_weight=smooth_weight,
                            )
                        )
                        color_index = (color_index + 1) % len(COLORS)
                    linestyle_index = (linestyle_index + 1) % len(LINE_STYLES)
                    color_index = 0

                plt.xlabel("train epochs")
                plt.ylabel("metric value")
                plt.legend(
                    handles=curves,
                    labels=labels,
                    loc="upper center",
                    bbox_to_anchor=(0.5, -0.05),
                    fancybox=True,
                    shadow=True,
                    ncol=2,
                )
                plt.tight_layout()
                plt.savefig(
                    save_path + f"fix_click_model/{dataset}/{metric}({click_type}).png"
                )
                # plt.show()
                plt.clf()

                color_index, linestyle_index = 0, 0  # reset indexes


def plot_with_fixed_user_model(
    data_path,
    save_path,
    datasets,
    alg_types,
    model_types,
    click_types,
):
    print("Each plot with fixed user model.")

    metrics = [
        "Validation_mrr_3",
        "Validation_mrr_5",
        "Validation_mrr_10",
        "Validation_ndcg_3",
        "Validation_ndcg_5",
        "Validation_ndcg_10",
    ]
    folds = list(range(1, 2))
    # runs = list(range(1, 4))
    runs = list(range(1, 2))

    num_iterations = 10000
    # num_iterations = 50000
    # smooth_weight = 0.9
    smooth_weight = 0.1

    for dataset in datasets:  # for each dataset
        if not os.path.exists(save_path + f"fix_user_model/{dataset}/"):
            os.makedirs(save_path + f"fix_user_model/{dataset}/")
        for metric in metrics:  # for each metric
            color_index = 0
            linestyle_index = 0  # indicate different model_type
            for mode_type in model_types:
                labels, curves = [], []
                plt.title(f"{metric} ({dataset}, {mode_type})")
                for click_type in click_types:
                    for alg in alg_types:  # for each alg
                        labels.append(f"{alg}_{click_type}")
                        curves.append(
                            plot_one_curve(
                                path=data_path + f"{dataset}",
                                alg_name=alg,
                                click_type=click_type,
                                mode_type=mode_type,
                                metric_type=metric,
                                folds=folds,
                                runs=runs,
                                num_iterations=num_iterations,
                                color=color_index,
                                line_style=linestyle_index,
                                smooth_weight=smooth_weight,
                            )
                        )
                        color_index = (color_index + 1) % len(COLORS)
                    linestyle_index = (linestyle_index + 1) % len(LINE_STYLES)
                    color_index = 0
                plt.xlabel("train epochs")
                plt.ylabel("metric value")
                plt.legend(
                    handles=curves,
                    labels=labels,
                    loc="upper center",
                    bbox_to_anchor=(0.5, -0.05),
                    fancybox=True,
                    shadow=True,
                    ncol=2,
                )
                plt.tight_layout()
                plt.savefig(
                    save_path + f"fix_user_model/{dataset}/{metric}({mode_type}).png"
                )
                # plt.show()
                plt.clf()

                color_index, linestyle_index = 0, 0  # reset indexes


def plot_with_fixed_user_model_and_click_model(
    data_path,
    save_path,
    datasets,
    alg_types,
    model_types,
    click_types,
):
    print("Each plot with fixed user model and click model.")

    metrics = [
        "Validation_mrr_3",
        "Validation_mrr_5",
        "Validation_mrr_10",
        "Validation_ndcg_3",
        "Validation_ndcg_5",
        "Validation_ndcg_10",
    ]
    folds = list(range(1, 2))
    # runs = list(range(1, 4))
    runs = list(range(1, 2))

    num_iterations = 10000
    # num_iterations = 50000
    # smooth_weight = 0.9
    smooth_weight = 0.1

    for dataset in datasets:  # for each dataset
        if not os.path.exists(save_path + f"fix_user_and_click_model/{dataset}/"):
            os.makedirs(save_path + f"fix_user_and_click_model/{dataset}/")
        for metric in metrics:  # for each metric
            color_index = 0
            for mode_type in model_types:
                for click_type in click_types:
                    labels, curves = [], []
                    plt.title(f"{metric} ({dataset}, {mode_type}, {click_type})")
                    for alg in alg_types:  # for each alg
                        labels.append(f"{alg}")
                        curves.append(
                            plot_one_curve(
                                path=data_path + f"{dataset}",
                                alg_name=alg,
                                click_type=click_type,
                                mode_type=mode_type,
                                metric_type=metric,
                                folds=folds,
                                runs=runs,
                                num_iterations=num_iterations,
                                color=color_index,
                                line_style=0,
                                smooth_weight=smooth_weight,
                            )
                        )
                        color_index = (color_index + 1) % len(COLORS)
                    plt.xlabel("train epochs")
                    plt.ylabel("metric value")
                    plt.legend(
                        handles=curves,
                        labels=labels,
                        loc="upper center",
                        bbox_to_anchor=(0.5, -0.05),
                        fancybox=True,
                        shadow=True,
                        ncol=2,
                    )
                    plt.tight_layout()
                    plt.savefig(
                        save_path
                        + f"fix_user_and_click_model/{dataset}/{metric}({mode_type}, {click_type}).png"
                    )
                    # plt.show()
                    plt.clf()

                    color_index = 0  # reset indexes


if __name__ == "__main__":
    path = whole_path + "results_tune_rnn/"
    save_path = whole_path + "plots/"

    # datasets = ['MQ2008', 'MSLRWEB10K']
    # datasets = ["MQ2008"]
    datasets = ["MSLRWEB10K"]

    # alg_types = ["BCQ", "DQN", "DoubleDQN", "DLA"]
    # alg_types = ["DQN", "DoubleDQN", "Bandit", "DLA"]
    alg_types = [
        "DLA",
        # "Bandit",
        # "DQN_avg",
        # "DQN_position",
        # "DQN_position_avg",
        # "DQN_position_avg_rew",
        # "DQN_avg_rew",
        # "DQN_rew",
        # "DoubleDQN_avg",
        # "DoubleDQN_position",
        # "DoubleDQN_position_avg",
        # "DoubleDQN_position_avg_rew",
        # "DoubleDQN_avg_rew",
        # "DoubleDQN_rew",
        # "CQL_avg",
        # "CQL_position",
        # "CQL_position_avg",
        # "CQL_position_avg_rew",
        # "CQL_avg_rew",
        # "CQL_rew",
        "CQL_avg_256",
        "CQL_avg_rew_256",
        "CQL_rew_256",
        "CQL_avg_512",
        "CQL_avg_rew_512",
        "CQL_rew_512",
    ]

    # model_types = ["navigational", "informational", "perfect"]
    # model_types = ["informational", "perfect"]
    model_types = ["informational"]
    # click_types = ["pbm", "cascade"]
    click_types = ["cascade"]

    ## plot with fixed click model
    plot_with_fixed_click_model(
        data_path=path,
        save_path=save_path,
        datasets=datasets,
        alg_types=alg_types,
        model_types=model_types,
        click_types=click_types,
    )
    # plot with fixed user model
    plot_with_fixed_user_model(
        data_path=path,
        save_path=save_path,
        datasets=datasets,
        alg_types=alg_types,
        model_types=model_types,
        click_types=click_types,
    )
    # plot with fixed user and click model
    plot_with_fixed_user_model_and_click_model(
        data_path=path,
        save_path=save_path,
        datasets=datasets,
        alg_types=alg_types,
        model_types=model_types,
        click_types=click_types,
    )
