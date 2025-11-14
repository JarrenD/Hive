import pytest
import torch
from hivegame.hivenet_td import make_value_model
from hivegame.hivenet_mcts import MCTSWithValue
from hivegame.hive import Hive

def test_mcts_run_and_select_move():
    model = make_value_model(torch.device("cpu"))
    mcts = MCTSWithValue(model, max_nodes=10, candidate_k=4)
    hive = Hive()
    hive.turn += 1
    hive.setup()
    color = "w"
    turn = hive.turn
    stats = mcts.run(hive, color, turn)
    assert "visits" in stats
    assert "q_values" in stats
    assert "best_move_cmd" in stats
    assert isinstance(stats["visits"], dict)
    assert isinstance(stats["q_values"], dict)
    assert isinstance(stats["best_move_cmd"], str)
    # Should not return 'pass' if legal moves exist
    if len(stats["visits"]) > 0:
        assert not (stats["best_move_cmd"] == "pass" and len(stats["visits"]) > 1)