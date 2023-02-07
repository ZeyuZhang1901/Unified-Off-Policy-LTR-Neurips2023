import torch


class AbstractRanker:
    def __init__(
        self,
        hyper_json_file,  # str, hyper params json file for ranker
        feature_size,
        rank_list_size,
        max_visuable_size,
        click_model,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.feature_size = feature_size
        self.rank_list_size = rank_list_size
        self.max_visuable_size = max_visuable_size
        self.click_model = click_model
