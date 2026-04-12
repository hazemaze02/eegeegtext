import os

PROJECT_NAME = "sentence_decoding"

SLURM_PARTITION = os.getenv("SLURM_PARTITION", "")
DATADIR = os.getenv("DATAPATH", "")
BASEDIR = os.getenv("SAVEPATH", "")
CACHEDIR = os.path.join(BASEDIR, "cache", PROJECT_NAME)
SAVEDIR = os.path.join(BASEDIR, "results", PROJECT_NAME)
NUM_CPUS = 8

default_config = {
    "seed": 0,
    "use_wandb": True,
    "cache": CACHEDIR,
    "project": PROJECT_NAME,
    "infra": {
        "cluster": "auto",
        "folder": SAVEDIR,
        "slurm_partition": SLURM_PARTITION,
        "timeout_min": 60 * 24 * 3,
        "gpus_per_node": 1,
        "cpus_per_task": NUM_CPUS,
        "job_name": "sentence_decoding",
        "workdir": {
            "copied": ["neuralset", "neuraltrain", "projects"],
            "includes": ["*.py"],
        },
    },
    "trainer_config": {
        "n_epochs": 50,  # 200
        "transformer_start_epoch": 0,
        "monitor": "val_retrieval_acc10_size=all_macro_0",
        "patience": 10,
        "lr": 1e-4,
        "fast_dev_run": False,
    },
    "data": {
        "cache": CACHEDIR,
        "data_path": DATADIR,
        "n_timelines": 10000,
        "neuro": {
            "name": "Meg",
            "scaler": "RobustScaler",
            "frequency": 50.0,
            "filter": (0.1, 40.0),
            "baseline": (0.0, 0.5),
            "offset": 0,
            "aggregation": "average",  # XXX used for grouped meg
            "clamp": 5,
            "infra": {
                "keep_in_ram": True,
                "folder": CACHEDIR,
                "cluster": None,
                "max_jobs": 10,
            },
        },
        "feature": {
            "name": "HuggingFaceText",
            # "model_name": "facebook/opt-2.7b",
            "model_name": "t5-large",
            "aggregation": "trigger",
            "layers": 0.5,
            "infra": {
                "keep_in_ram": True,
                "folder": CACHEDIR,
                "cluster": None,
            },
            "device": "cpu",
        },
        "start": 0.0,
        "duration": 3.0,
        "batch_size": 128,
        "num_workers": NUM_CPUS,
    },
    "brain_model_config": {
        "name": "SimpleConvTimeAgg",
        "time_agg_out": "att",
        "dropout_input": 0.1,
        "hidden": 160,
        "batch_norm": True,
        "depth": 5,
        "dilation_period": 5,
        "kernel_size": 3,
        "skip": True,
        "subject_layers": True,
        "complex_out": False,
        "glu": 2,
        "glu_context": 1,
        "merger": True,
        "initial_linear": 512,
        "gelu": True,
        "merger_pos_dim": 2048,
        "merger_per_subject": True,
        "n_subjects": 500,
    },
    "use_transformer": True,
    "use_target_scaler": False,
    "transformer_config": {"name": "TransformerEncoder", "depth": 16, "heads": 16},
    "loss": {
        "name": "SigLip",
    },
    "metrics": [
        {
            "log_name": "contrastive_cosine_sim",
            "name": "CosineSimilarity",
            "kwargs": {"reduction": "mean"},
        },
        {"log_name": "contrastive_median_rank", "name": "Rank", "reduction": "median"},
        {"log_name": "contrastive_mean_rank", "name": "Rank", "reduction": "mean"},
        {"log_name": "contrastive_top1_acc", "name": "TopkAcc", "topk": "1"},
        {"log_name": "contrastive_top5_acc", "name": "TopkAcc", "topk": "5"},
    ],
    "retrieval_metrics": [
        {
            "log_name": "retrieval_rank",
            "name": "Rank",
            "reduction": "median",
        },
        {
            "log_name": "retrieval_rank_instance-agg",
            "name": "Rank",
            "reduction": "median",
        },
        {"log_name": "retrieval_acc1", "name": "TopkAcc", "topk": 1},
        {
            "log_name": "retrieval_acc1_instance-agg",
            "name": "TopkAcc",
            "topk": 1,
        },
        {"log_name": "retrieval_acc10", "name": "TopkAcc", "topk": 10},
        {
            "log_name": "retrieval_acc10_instance-agg",
            "name": "TopkAcc",
            "topk": 10,
        },
        {"log_name": "sentence_bleu1", "name": "BLEUScore", "kwargs": {"n_gram": 1}},
        {"log_name": "sentence_bleu2", "name": "BLEUScore", "kwargs": {"n_gram": 2}},
        {"log_name": "wer", "name": "WordErrorRate"},
        {
            "log_name": "sentence_bert",
            "name": "BERTScore",
            "kwargs": {"model_name_or_path": "bert-base-uncased"},
        },
    ],
}
