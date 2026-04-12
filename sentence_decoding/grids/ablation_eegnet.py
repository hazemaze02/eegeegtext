"""Grid over different configurations.
"""

from neuraltrain.utils import run_grid, update_config
from sentence_decoding.main import Experiment

from .defaults import default_config  # type: ignore

GRID_NAME = "ablation_eegnet"

update = {
    "infra.job_name": GRID_NAME,
    "data.dataset": "Armeni2022",
    "data.feature.name": "HuggingFaceText",
    "data.event_type": "Word",
    "brain_model_config": {"name": "EEGNet"},
    "use_transformer": False,
}
updated_config = update_config(default_config, update)

grid = {
    "data.dataset": [
        "Broderick2019",
        "Armeni2022",
        "Gwilliams2022",
        "PallierListen2023",
        "PallierRead2023",
        "Nieuwland2018",
        "Accou2023",
        "SchoffelenRead2019",
        "SchoffelenListen2019",
    ]
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
        infra_mode="retry",
    )
