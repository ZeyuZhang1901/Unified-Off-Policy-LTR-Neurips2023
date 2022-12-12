import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

from ranker.AbstractRanker import AbstractRanker
from network.CQL import Actor, Critic
from collections import namedtuple

Transition = namedtuple(
    "Transition", ("state", "action", "next_state", "reward", "done", "chosen", "qid")
)

class CQLRanker(AbstractRanker):
    def 
