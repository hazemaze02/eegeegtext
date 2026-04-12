"""Grid over different configurations.
"""

from neuraltrain.utils import run_grid, update_config
from sentence_decoding.main import Experiment

from .defaults import default_config  # type: ignore

GRID_NAME = "ablation_duration"

update = {
    "infra.job_name": GRID_NAME,
    "data.dataset": "Armeni2022",
    "data.neuro.baseline": (0, 0.1),
}
updated_config = update_config(default_config, update)

grid = {
    "data.duration": [0.1, 0.2, 0.3, 0.4, 0.5, 1, 2, 3],
    "data.dataset": [
        "Nieuwland2018",
        "Accou2023",
        "SchoffelenRead2019",
        "SchoffelenListen2019",
        # "Broderick2019",
        # "Gwilliams2022",
        # "Armeni2022",
        # "PallierListen2023",
        # "PallierRead2023",
    ],
}

if __name__ == "__main__":
    out = run_grid(
        Experiment,
        GRID_NAME,
        updated_config,
        grid,
        combinatorial=True,
        overwrite=False,
        dry_run=False,
    )
