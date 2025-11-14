import pytest
import torch
from hivegame.hivenet_td import make_value_model
from hivegame.hivenet_mcts import MCTSWithValue
from hivegame.hive import Hive

class DummyModel(torch.nn.Module):
    def forward(self, x, reserve_vec):
        # Always return +1 for first batch, -1 for others
        return torch.ones(x.shape[0])

def test_mcts_value_sign():
    model = DummyModel()
    mcts = MCTSWithValue(model, max_nodes=5, candidate_k=2)
    hive = Hive()
    hive.turn += 1
    hive.setup()
    color = "w"
    turn = hive.turn
    stats = mcts.run(hive, color, turn)
    # Should pick a move and Q at root should be positive
    assert stats["best_move_cmd"] != "pass"
    assert all(q >= 0 for q in stats["q_values"].values())