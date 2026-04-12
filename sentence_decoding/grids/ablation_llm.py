"""Grid over different configurations.
"""

from neuraltrain.utils import run_grid, update_config
from sentence_decoding.main import Experiment

from .defaults import default_config  # type: ignore

GRID_NAME = "ablation_llm"

update = {
    "infra.job_name": GRID_NAME,
    "data.dataset": "Armeni2022",
    "data.feature.name": "HuggingFaceText",
    "data.event_type": "Word",
}
updated_config = update_config(default_config, update)

grid = {
    "data.feature.model_name": [
        "google/mt5-large",
        # "facebook/opt-1.3b",
        # ""
        # "google-t5/t5-small",
        # "google-t5/t5-base",
        # "google-t5/t5-large",
        # "google-t5/t5-3b",
        # "google-t5/t5-11b",
        # "gpt2",
        # "gpt2-medium",
        # "gpt2-large",
        # "bert-base-uncased",
    ],
    "data.dataset": [
        "Nieuwland2018",
        "Accou2023",
        "SchoffelenRead2019",
        "SchoffelenListen2019",
        "Broderick2019",
        "Gwilliams2022",
        "Armeni2022",
        "PallierListen2023",
        "PallierRead2023",
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
