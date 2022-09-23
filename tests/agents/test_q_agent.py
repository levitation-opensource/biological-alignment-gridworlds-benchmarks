import pytest

from aintelope.aintelope.training.simple_eval import run_episode


def test_qagent_in_savanna_zoo_sequential():
    # get the default params from training.lightning.yaml
    # then override with these test params

    # TODO: refactor out into test constants? Or leave here? /shrug
    hparams = {
        "agent": "q_agent",
        "env": "savanna-zoo-v2",
        "env_entry_point": None,
        "env_type": "zoo",
        "env_params": {
            "NUM_ITERS": 40,  # duration of the game
            "MAP_MIN": 0,
            "MAP_MAX": 20,
            "render_map_max": 20,
            "AMOUNT_AGENTS": 1,  # for now only one agent
            "AMOUNT_GRASS_PATCHES": 2,
        },
    }

    run_episode(hparams=hparams)
