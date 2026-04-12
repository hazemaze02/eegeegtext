"""Grid over different configurations.
"""

from neuraltrain.utils import run_grid, update_config
from sentence_decoding.main import Experiment

from .defaults import default_config  # type: ignore

GRID_NAME = "paper_final"

update = {
    "infra.job_name": GRID_NAME,
    "use_transformer": True,
    "use_target_scaler": False,
    # "data.feature.model_name": "google/mt5-large",
    # "data.feature": {"name":"FastTextEmbedding", "model_folder": "/data/home/sdascoli/word2vec", "aggregation": "trigger", "infra": {"folder": default_config["infra"]["folder"]}},
    # "transformer_config.heads": 6,
}
updated_config = update_config(default_config, update)

grid = {
    "data.dataset": [
        # "Nieuwland2018",
        "Accou2023",
        # "SchoffelenRead2019",
        # "SchoffelenListen2019",
        # "Armeni2022",
        # "Broderick2019",
        # "Gwilliams2022",
        # "PallierListen2023",
        # "PallierRead2023",
    ],
    # "data.n_timelines": [500, 1000],
    # "data.feature.model_name": ["google/mt5-large"],
    #     "t5-large",
    # "facebook/opt-1.3b",
    # "facebook/opt-2.7b",
    # "facebook/opt-6.7b",
    # ],
    # "loss.name": ["Clip", "SigLip"],
    # "use_target_scaler": [True, False],
    "use_transformer": [True, False],
    # "trainer_config.noise_ratio": [0.0, 0.5, 2],
    # "trainer_config.erasing_ratio": [0.0, 0.75, 0.9],
    # "use_transformer": [True, False],
    # "transformer_config.depth": [20, 24, 28, 32, 36],
    # "transformer_start_epoch": [0, 5, 10],
    # "data.min_sentence_duration": [3],
}

if __name__ == "__main__":
    out = run_grid(
        Experiment,
        GRID_NAME,
        updated_config,
        grid,
        combinatorial=True,
        overwrite=True,
        dry_run=False,
    )
