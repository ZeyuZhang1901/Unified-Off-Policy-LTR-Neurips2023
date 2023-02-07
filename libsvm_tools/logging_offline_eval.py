import os
import sys
import torch

metric_type = ["ndcg", "err"]
# metric_type = ["ndcg"]
metric_position = [3, 5, 10]
max_length = 0
# max_label = 2  # or 4 for five fold dataset
max_label = 4  # or 4 for five fold dataset


def safe_div(numerator, denominator):
    return torch.where(
        torch.eq(denominator, 0),
        torch.zeros_like(numerator),
        torch.div(numerator, denominator),
    )


def read_one_line(label_fin):
    label_line = label_fin.readline()
    if label_line == "":
        return None, None
    arr = label_line.strip().split(" ")
    qid = arr[0]
    initial_label_list = [float(i) for i in arr[1:]]

    return qid, initial_label_list


def prepare_one_label_set(label_path, output_path, set_name):
    global metric_position, metric_type
    global max_length
    global max_label
    qid_list = []
    label_fin = open(label_path + set_name + ".labels")

    ## read all label lists
    print(f"Read {set_name} input label lists...")
    label_lists = []
    qid, label_list = read_one_line(label_fin)
    while qid is not None:
        qid_list.append(qid)
        label_lists.append(label_list)
        max_length = max(len(label_list), max_length)
        qid, label_list = read_one_line(label_fin)
    label_fin.close()

    ## pad with zero
    print("Padding...")
    for i in range(len(label_lists)):
        label_lists[i] += [0.0 for _ in range(max_length - len(label_lists[i]))]

    ## calculate metric for each initial rank
    print("Calculate metrics...")
    metric_map = {}  # e.g. {"ndcg": {3: 0.3, 5: 0.5, 10: 0.7}}
    labels_tensor = torch.tensor(label_lists)
    for metric in metric_type:
        if metric == "ndcg":
            discount = 1 / torch.log2(torch.arange(max_length) + 2)
            dcg = (torch.pow(2, labels_tensor) - 1) * discount
            dcg = torch.cumsum(dcg, dim=1)
            max_dcg = (
                torch.pow(2, labels_tensor.sort(descending=True, dim=1)[0]) - 1
            ) * discount
            max_dcg = torch.cumsum(max_dcg, dim=1)
            ndcg = safe_div(dcg, max_dcg).mean(dim=0)
            metric_map[metric] = {}
            for pos in metric_position:
                metric_map[metric][pos] = ndcg[pos-1].item()
        elif metric == "err":
            relevance = (torch.pow(2, labels_tensor) - 1) / pow(2, max_label)
            non_rel = torch.cumprod(1.0 - relevance, dim=1) / (1.0 - relevance)
            reciprocal_rank = 1.0 / torch.arange(start=1, end=max_length + 1)
            mask = [
                torch.ge(reciprocal_rank, 1.0 / n).type(torch.float32)
                for n in metric_position
            ]
            reciprocal_rank_topn = [reciprocal_rank * top_n_mask for top_n_mask in mask]
            # ERR has a shape of [batch_size, 1]
            err = [
                torch.sum(
                    relevance * non_rel * reciprocal_rank,
                    dim=1,
                    keepdim=True,
                )
                for reciprocal_rank in reciprocal_rank_topn
            ]
            err = torch.stack(err, dim=0).mean(dim=1)
            metric_map[metric] = {}
            i = 0
            for pos in metric_position:
                metric_map[metric][pos] = err[i].item()
                i += 1

    ## write metric
    print("Write to metric file")
    metric_fout = open(output_path + set_name + ".metric", "w")
    line = ""
    for metric in metric_type:
        line += " ".join(
            [f"{metric}_{pos}:{value}" for pos, value in metric_map[metric].items()]
        )
        line += "\n"
    metric_fout.write(line)

    metric_fout.close()


def main():
    DATA_PATH = sys.argv[1]  # for svm, point to `tmp_data` folder
    OUTPUT_PATH = sys.argv[2]  # for svm, point to `tmp_data` folder
    SET_NAME = ["train", "test", "valid"]

    print(f"Metric Type: {' '.join(metric for metric in metric_type)}")
    print(f"Position: {' '.join(str(pos) for pos in metric_position)}")
    for set_name in SET_NAME:
        if not os.path.exists(OUTPUT_PATH + set_name + "/"):
            os.makedirs(OUTPUT_PATH + set_name + "/")
        prepare_one_label_set(
            DATA_PATH + set_name + "/",
            OUTPUT_PATH + set_name + "/",
            set_name,
        )


if __name__ == "__main__":
    main()
