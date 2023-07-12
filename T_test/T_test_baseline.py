import scipy
import numpy as np


def t_test(data_tuple1, data_tuple2):
    """Perform t-test on two data tuples.

    Args:
        data_tuple1: tuple, A tuple of data.
        data_tuple2: tuple, A tuple of data.
        alpha: The significance level.

    Returns:
        A tuple of (t, p), where t is the t-statistics and p is the p-value.
    """
    t, p = scipy.stats.ttest_rel(data_tuple1, data_tuple2)
    return t, p


def read_data_from_logs(log_file):
    """Read data from log files.

    Args:
        log_file: str, The path to the log file.
        metric_name: str, The name of the metric.

    Returns:
        A tuple of data.
    """
    data = {
        "pbm": {
            "DLA": [],
            "CM_IPW": [],
            "IPW": [],
            "ATTENTION_CQL": [],
            "ATTENTION_SAC": [],
        },
        "dcm": {
            "DLA": [],
            "CM_IPW": [],
            "IPW": [],
            "ATTENTION_CQL": [],
            "ATTENTION_SAC": [],
        },
        "cascade": {
            "DLA": [],
            "CM_IPW": [],
            "IPW": [],
            "ATTENTION_CQL": [],
            "ATTENTION_SAC": [],
        },
        "ccm": {
            "DLA": [],
            "CM_IPW": [],
            "IPW": [],
            "ATTENTION_CQL": [],
            "ATTENTION_SAC": [],
        },
        "ubm": {
            "DLA": [],
            "CM_IPW": [],
            "IPW": [],
            "ATTENTION_CQL": [],
            "ATTENTION_SAC": [],
        },
    }
    with open(log_file, "r") as f:
        lines = f.readlines()
        for line in lines[1:]:
            line = line.strip().split()
            if line[0] == "mean":
                continue
            if line[2] == "pbm":
                if line[3] == "DLA":
                    data["pbm"]["DLA"].append([float(x) for x in line[4:]])
                elif line[3] == "CM_IPW":
                    data["pbm"]["CM_IPW"].append([float(x) for x in line[4:]])
                elif line[3] == "IPW":
                    data["pbm"]["IPW"].append([float(x) for x in line[4:]])
                elif line[3] == "ATTENTION_CQL":
                    data["pbm"]["ATTENTION_CQL"].append([float(x) for x in line[4:]])
                elif line[3] == "ATTENTION_SAC":
                    data["pbm"]["ATTENTION_SAC"].append([float(x) for x in line[4:]])
            elif line[2] == "dcm":
                if line[3] == "DLA":
                    data["dcm"]["DLA"].append([float(x) for x in line[4:]])
                elif line[3] == "CM_IPW":
                    data["dcm"]["CM_IPW"].append([float(x) for x in line[4:]])
                elif line[3] == "IPW":
                    data["dcm"]["IPW"].append([float(x) for x in line[4:]])
                elif line[3] == "ATTENTION_CQL":
                    data["dcm"]["ATTENTION_CQL"].append([float(x) for x in line[4:]])
                elif line[3] == "ATTENTION_SAC":
                    data["dcm"]["ATTENTION_SAC"].append([float(x) for x in line[4:]])
            elif line[2] == "cascade":
                if line[3] == "DLA":
                    data["cascade"]["DLA"].append([float(x) for x in line[4:]])
                elif line[3] == "CM_IPW":
                    data["cascade"]["CM_IPW"].append([float(x) for x in line[4:]])
                elif line[3] == "IPW":
                    data["cascade"]["IPW"].append([float(x) for x in line[4:]])
                elif line[3] == "ATTENTION_CQL":
                    data["cascade"]["ATTENTION_CQL"].append(
                        [float(x) for x in line[4:]]
                    )
                elif line[3] == "ATTENTION_SAC":
                    data["cascade"]["ATTENTION_SAC"].append(
                        [float(x) for x in line[4:]]
                    )
            elif line[2] == "ccm":
                if line[3] == "DLA":
                    data["ccm"]["DLA"].append([float(x) for x in line[4:]])
                elif line[3] == "CM_IPW":
                    data["ccm"]["CM_IPW"].append([float(x) for x in line[4:]])
                elif line[3] == "IPW":
                    data["ccm"]["IPW"].append([float(x) for x in line[4:]])
                elif line[3] == "ATTENTION_CQL":
                    data["ccm"]["ATTENTION_CQL"].append([float(x) for x in line[4:]])
                elif line[3] == "ATTENTION_SAC":
                    data["ccm"]["ATTENTION_SAC"].append([float(x) for x in line[4:]])
            elif line[2] == "ubm":
                if line[3] == "DLA":
                    data["ubm"]["DLA"].append([float(x) for x in line[4:]])
                elif line[3] == "CM_IPW":
                    data["ubm"]["CM_IPW"].append([float(x) for x in line[4:]])
                elif line[3] == "IPW":
                    data["ubm"]["IPW"].append([float(x) for x in line[4:]])
                elif line[3] == "ATTENTION_CQL":
                    data["ubm"]["ATTENTION_CQL"].append([float(x) for x in line[4:]])
                elif line[3] == "ATTENTION_SAC":
                    data["ubm"]["ATTENTION_SAC"].append([float(x) for x in line[4:]])

    return data


if __name__ == "__main__":
    # result_path = "/Users/zeyuzhang/Projects/results/metric_web10k_baselines.txt"
    # result_path = "/Users/zeyuzhang/Downloads/results_web10k_baseline_fold1_5/metric_baselines.txt"
    # result_path = (
    #     "/Users/zeyuzhang/Projects/results/results_yahoo/metric_yahoo_baselines.txt"
    # )
    result_path = "/Users/zeyuzhang/Projects/results/results_web10k/metric_baselines.txt"

    # output_path = "/Users/zeyuzhang/Projects/results/metric_web10k_baselines_process.txt"
    # result_path = "/Users/zeyuzhang/Downloads/results_web10k_baseline_fold1_5/metric_baselines_process.txt"
    # output_path = "/Users/zeyuzhang/Projects/results/results_yahoo/metric_yahoo_baselines_process.txt"
    output_path = "/Users/zeyuzhang/Projects/results/results_web10k/metric_baselines_process.txt"

    print("data reading start!")
    data = read_data_from_logs(result_path)
    print("data reading finish!")

    fout = open(output_path, "w")

    ## t-test
    print("t-test start!")
    fout.write("T-TEST:\n")
    for click_type in ["pbm", "dcm", "cascade", "ccm", "ubm"]:
        # for click_type in ["pbm", "cascade"]:
        base_performance = data[click_type]["ATTENTION_SAC"]
        print(f"base_algorithm: ATTENTION_SAC")
        fout.write(f"base_algorithm: ATTENTION_SAC\n\n")
        print(f"********click type: {click_type}**********")
        fout.write(f"********click type: {click_type}**********\n")
        for exp in ["DLA", "CM_IPW", "IPW"]:
            exp_performance = data[click_type][exp]
            print(f"exp_algorithm: {exp}")
            fout.write(f"exp_algorithm: {exp}\n")
            for i in range(6):  # for different metrics
                t, p = t_test(
                    np.array([performance[i] for performance in base_performance]),
                    np.array([performance[i] for performance in exp_performance]),
                )
                print(f"statistic: {t}, p-value: {p}")
                fout.write(f"statistic: {t}, p-value: {p}\n")
            print("-" * 20)
            fout.write("-" * 20 + "\n")
        print("=" * 20 + "\n")
        fout.write("=" * 20 + "\n")

    fout.write("\n\n\n")

    for click_type in ["pbm", "dcm", "cascade", "ccm", "ubm"]:
        # for click_type in ["dcm", "ccm"]:
        base_performance = data[click_type]["ATTENTION_CQL"]
        print(f"base_algorithm: ATTENTION_CQL")
        fout.write(f"base_algorithm: ATTENTION_CQL\n\n")
        print(f"********click type: {click_type}**********")
        fout.write(f"********click type: {click_type}**********\n")
        for exp in ["DLA", "CM_IPW", "IPW"]:
            exp_performance = data[click_type][exp]
            print(f"exp_algorithm: {exp}")
            fout.write(f"exp_algorithm: {exp}\n")
            for i in range(6):  # for different metrics
                t, p = t_test(
                    np.array([performance[i] for performance in base_performance]),
                    np.array([performance[i] for performance in exp_performance]),
                )
                print(f"statistic: {t}, p-value: {p}")
                fout.write(f"statistic: {t}, p-value: {p}\n")
            print("-" * 20)
            fout.write("-" * 20 + "\n")
        print("=" * 20 + "\n")
        fout.write("=" * 20 + "\n")
    print("t-test finish!")

    # ## mean metric
    # print("mean metric start!")
    # fout.write("MEAN METRIC:\n")
    # for click_type in ["pbm", "dcm", "cascade", "ccm"]:
    #     print(f"********click type: {click_type}**********")
    #     fout.write(f"********click type: {click_type}**********\n")
    #     for exp in ["DLA", "CM_IPW", "IPW", "ATTENTION_CQL", "ATTENTION_SAC"]:
    #         print(f"exp_algorithm: {exp}")
    #         fout.write(f"exp_algorithm: {exp}\n")
    #         for i in range(6):
    #             print(
    #                 f"metric {i}: {np.mean([performance[i] for performance in data[click_type][exp]])}"
    #             )
    #             fout.write(
    #                 f"metric {i}: {np.mean([performance[i] for performance in data[click_type][exp]])}\n"
    #             )
    #         print("-" * 20)
    #         fout.write("-" * 20 + "\n")
    #     print("=" * 20 + "\n")
    #     fout.write("=" * 20 + "\n")
    # print("mean metric finish!")

    fout.close()
