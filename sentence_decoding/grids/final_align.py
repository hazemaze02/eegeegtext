"""Grid over different configurations.
"""

from neuraltrain.utils import run_grid, update_config
from sentence_decoding.alignment import DecodingExperiment as Experiment

from .defaults import default_config  # type: ignore

GRID_NAME = "paper_align"

update = {"infra.job_name": GRID_NAME, "mapper": {"stride": 1}, "data.start": -0.5}
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
