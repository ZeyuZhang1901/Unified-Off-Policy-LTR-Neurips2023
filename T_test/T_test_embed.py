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
            "position": [],
            "predoc": [],
            "both": [],
            "RNN": [],
            "LSTM": [],
            "ATTENTION": [],
        },
        "dcm": {
            "position": [],
            "predoc": [],
            "both": [],
            "RNN": [],
            "LSTM": [],
            "ATTENTION": [],
        },
        "cascade": {
            "position": [],
            "predoc": [],
            "both": [],
            "RNN": [],
            "LSTM": [],
            "ATTENTION": [],
        },
        "ccm": {
            "position": [],
            "predoc": [],
            "both": [],
            "RNN": [],
            "LSTM": [],
            "ATTENTION": [],
        },
    }
    with open(log_file, "r") as f:
        lines = f.readlines()
        for line in lines[1:]:
            line = line.strip().split()
            if line[0] == "mean":
                continue
            if line[1] == "pbm":
                if line[2] == "position":
                    data["pbm"]["position"].append([float(x) for x in line[3:]])
                elif line[2] == "predoc":
                    data["pbm"]["predoc"].append([float(x) for x in line[3:]])
                elif line[2] == "both":
                    data["pbm"]["both"].append([float(x) for x in line[3:]])
                elif line[2] == "RNN":
                    data["pbm"]["RNN"].append([float(x) for x in line[3:]])
                elif line[2] == "LSTM":
                    data["pbm"]["LSTM"].append([float(x) for x in line[3:]])
                elif line[2] == "ATTENTION":
                    data["pbm"]["ATTENTION"].append([float(x) for x in line[3:]])
            elif line[1] == "dcm":
                if line[2] == "position":
                    data["dcm"]["position"].append([float(x) for x in line[3:]])
                elif line[2] == "predoc":
                    data["dcm"]["predoc"].append([float(x) for x in line[3:]])
                elif line[2] == "both":
                    data["dcm"]["both"].append([float(x) for x in line[3:]])
                elif line[2] == "RNN":
                    data["dcm"]["RNN"].append([float(x) for x in line[3:]])
                elif line[2] == "LSTM":
                    data["dcm"]["LSTM"].append([float(x) for x in line[3:]])
                elif line[2] == "ATTENTION":
                    data["dcm"]["ATTENTION"].append([float(x) for x in line[3:]])
            elif line[1] == "cascade":
                if line[2] == "position":
                    data["cascade"]["position"].append([float(x) for x in line[3:]])
                elif line[2] == "predoc":
                    data["cascade"]["predoc"].append([float(x) for x in line[3:]])
                elif line[2] == "both":
                    data["cascade"]["both"].append([float(x) for x in line[3:]])
                elif line[2] == "RNN":
                    data["cascade"]["RNN"].append([float(x) for x in line[3:]])
                elif line[2] == "LSTM":
                    data["cascade"]["LSTM"].append([float(x) for x in line[3:]])
                elif line[2] == "ATTENTION":
                    data["cascade"]["ATTENTION"].append([float(x) for x in line[3:]])
            elif line[1] == "ccm":
                if line[2] == "position":
                    data["ccm"]["position"].append([float(x) for x in line[3:]])
                elif line[2] == "predoc":
                    data["ccm"]["predoc"].append([float(x) for x in line[3:]])
                elif line[2] == "both":
                    data["ccm"]["both"].append([float(x) for x in line[3:]])
                elif line[2] == "RNN":
                    data["ccm"]["RNN"].append([float(x) for x in line[3:]])
                elif line[2] == "LSTM":
                    data["ccm"]["LSTM"].append([float(x) for x in line[3:]])
                elif line[2] == "ATTENTION":
                    data["ccm"]["ATTENTION"].append([float(x) for x in line[3:]])

    return data


if __name__ == "__main__":
    # result_path = "/Users/zeyuzhang/Projects/results/metric_embedding_yahoo.txt"
    # output_path = "/Users/zeyuzhang/Projects/results/metric_embedding_yahoo_process.txt"
    result_path = "/Users/zeyuzhang/Projects/results/metric_embedding_yahoo2.txt"
    output_path = "/Users/zeyuzhang/Projects/results/metric_embedding_yahoo_process2.txt"

    print("data reading start!")
    data = read_data_from_logs(result_path)
    print("data reading finish!")

    fout = open(output_path, "w")

    ## t-test
    print("t-test start!")
    fout.write("T-TEST:\n")
    # for click_type in ["pbm", "dcm", "cascade", "ccm"]:
    for click_type in ["pbm", "cascade"]:
        base_performance = data[click_type]["ATTENTION"]
        print(f"base_algorithm: ATTENTION")
        fout.write(f"base_algorithm: ATTENTION\n\n")
        print(f"********click type: {click_type}**********")
        fout.write(f"********click type: {click_type}**********\n")
        for exp in ["position", "predoc", "both"]:
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

    # for click_type in ["pbm", "dcm", "cascade", "ccm"]:
    #     base_performance = data[click_type]["LSTM"]
    #     print(f"base_algorithm: LSTM")
    #     fout.write(f"base_algorithm: LSTM\n\n")
    #     print(f"********click type: {click_type}**********")
    #     fout.write(f"********click type: {click_type}**********\n")
    #     for exp in ["position", "predoc", "both"]:
    #         exp_performance = data[click_type][exp]
    #         print(f"exp_algorithm: {exp}")
    #         fout.write(f"exp_algorithm: {exp}\n")
    #         for i in range(6):  # for different metrics
    #             t, p = t_test(
    #                 np.array([performance[i] for performance in base_performance]),
    #                 np.array([performance[i] for performance in exp_performance]),
    #             )
    #             print(f"statistic: {t}, p-value: {p}")
    #             fout.write(f"statistic: {t}, p-value: {p}\n")
    #         print("-" * 20)
    #         fout.write("-" * 20 + "\n")
    #     print("=" * 20 + "\n")
    #     fout.write("=" * 20 + "\n")
    # print("t-test finish!")

    # fout.write("\n\n\n")

    # for click_type in ["pbm", "dcm", "cascade", "ccm"]:
    #     base_performance = data[click_type]["RNN"]
    #     print(f"base_algorithm: RNN")
    #     fout.write(f"base_algorithm: RNN\n\n")
    #     print(f"********click type: {click_type}**********")
    #     fout.write(f"********click type: {click_type}**********\n")
    #     for exp in ["position", "predoc", "both"]:
    #         exp_performance = data[click_type][exp]
    #         print(f"exp_algorithm: {exp}")
    #         fout.write(f"exp_algorithm: {exp}\n")
    #         for i in range(6):  # for different metrics
    #             t, p = t_test(
    #                 np.array([performance[i] for performance in base_performance]),
    #                 np.array([performance[i] for performance in exp_performance]),
    #             )
    #             print(f"statistic: {t}, p-value: {p}")
    #             fout.write(f"statistic: {t}, p-value: {p}\n")
    #         print("-" * 20)
    #         fout.write("-" * 20 + "\n")
    #     print("=" * 20 + "\n")
    #     fout.write("=" * 20 + "\n")
    # print("t-test finish!")

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
