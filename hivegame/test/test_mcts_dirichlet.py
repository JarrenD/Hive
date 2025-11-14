import pytest
import torch
import numpy as np
from hivegame.hivenet_td import make_value_model
from hivegame.hivenet_mcts import MCTSWithValue
from hivegame.hive import Hive

def entropy(p):
    p = np.array(p)
    p = p[p > 0]
    return -np.sum(p * np.log(p))

def test_mcts_dirichlet_noise_effect():
    model = make_value_model(torch.device("cpu"))
    hive = Hive()
    hive.turn += 1
    hive.setup()
    color = "w"
    turn = hive.turn

    # With Dirichlet
    mcts1 = MCTSWithValue(model, max_nodes=10, candidate_k=4, add_dirichlet=True, dirichlet_alpha=0.5, dirichlet_frac=0.5)
    stats1 = mcts1.run(hive, color, turn)
    p1 = np.array(list(stats1["visits"].values()), dtype=np.float32)
    p1 = p1 / (p1.sum() if p1.sum() > 0 else 1)

    # Without Dirichlet
    mcts2 = MCTSWithValue(model, max_nodes=10, candidate_k=4, add_dirichlet=False)
    stats2 = mcts2.run(hive, color, turn)
    p2 = np.array(list(stats2["visits"].values()), dtype=np.float32)
    p2 = p2 / (p2.sum() if p2.sum() > 0 else 1)

    # Entropy should be higher with Dirichlet
    assert entropy(p1) >= entropy(p2) - 1e-3