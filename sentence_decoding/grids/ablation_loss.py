"""Grid over different configurations.
"""

from neuraltrain.utils import run_grid, update_config
from sentence_decoding.main import Experiment

from .defaults import default_config  # type: ignore

GRID_NAME = "ablation_loss"

update = {
    "infra.job_name": GRID_NAME,
}
updated_config = update_config(default_config, update)

grid = {
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
    "loss": [
        {"name": "Clip", "symmetric": True, "temperature": True, "norm_kind": "y"},
        {"name": "SigLip", "identical_candidates_threshold": None},
        {
            "name": "SigLip",
            "identical_candidates_threshold": 0.999,
            "reweigh_positives": False,
        },
        {
            "name": "SigLip",
            "identical_candidates_threshold": 0.999,
            "reweigh_positives": True,
        },
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
