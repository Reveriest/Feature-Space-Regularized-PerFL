import argparse
import copy
import datetime
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

from data import datasets
from utils import misc
from src.client import Client
from utils.hparams_registry import default_hparams
from utils.metrics import default_output_path, final_metrics
from src.server import Server
from utils.utils_infer import *


DATASETS = ["RotatedMNIST", "ColoredMNIST"]
BASE_AGGREGATORS = ["ours", "ours_1", "ours_12", "ours_13",
                    "local", "fedavg", "fedprox", "scaffold",
                    "fedproto", "fedrep", "pflda", "ditto",
                    "fedem", "fedwdp", "fedfocal"]
RWL_AGGREGATORS = ["local_rwl", "fedavg_rwl", "fedprox_rwl", "scaffold_rwl",
                   "fedrep_rwl", "pflda_rwl", "ditto_rwl", "fedwdp_rwl"]
EPOCHS = {
    "RotatedMNIST": {
        None: {"local": 150, "global": 5},
        0.005: {"local": 200, "global": 5},
    },
    "ColoredMNIST": {
        None: {"local": 150, "global": 4},
        0.005: {"local": 100, "global": 6},
    },
}


def build_eval_tag(heterogeneity, imbalance_ratio, samplesize_ratio):
    tags = []
    if heterogeneity is not None:
        tags.append(f"heter{heterogeneity}")
    if imbalance_ratio is not None:
        tags.append(f"im{str(imbalance_ratio).replace('.', '')}")
    if samplesize_ratio is not None:
        tags.append(f"sa{str(samplesize_ratio).replace('.', '')}")
    return "_".join(tags)


def set_random_seed(random_seed):
    os.environ["PYTHONHASHSEED"] = str(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)


def build_minority_map(train_datasets, dataset_name, num_classes):
    minority_count = 3 if dataset_name == "RotatedMNIST" else 1
    minority_map = {}

    for client_id, train_dataset in enumerate(train_datasets, start=1):
        class_counts = np.zeros(num_classes, dtype=int)

        for _, label in train_dataset:
            label_value = int(label.item()) if torch.is_tensor(label) else int(label)
            class_counts[label_value] += 1

        minority_labels = sorted(
            range(num_classes),
            key=lambda label: (class_counts[label], label),
        )[:minority_count]

        minority_map[client_id] = minority_labels

    return minority_map


def train_procedure(
    aggregator,
    dataset_name,
    trial_seed,
    local_epochs,
    global_epochs,
    heterogeneity=None,
    imbalance_ratio=None,
    samplesize_ratio=None,
    experiment_type="baseline",
):
    set_random_seed(trial_seed)
    conf = default_hparams(aggregator, dataset_name, trial_seed)

    if heterogeneity is not None:
        conf["heterogeneity"] = heterogeneity
    if imbalance_ratio is not None:
        conf["imbalance_ratio"] = imbalance_ratio
    if samplesize_ratio is not None:
        conf["samplesize_ratio"] = samplesize_ratio

    conf["local_epochs"] = local_epochs
    conf["global_epochs"] = global_epochs

    print("HParams:")
    for key, value in sorted(conf.items()):
        print("\t{}: {}".format(key, value))

    whole_dataset = vars(datasets)[dataset_name](conf["data_dir"], [-1], conf)
    print(
        f"Dataset {dataset_name} ----> Shape: {whole_dataset.input_shape}, Classes: {whole_dataset.num_classes}"
    )

    train_datasets = []
    calib_datasets = []
    test_datasets = []
    for environment_id, environment in enumerate(whole_dataset):
        train_dataset, calib_test_dataset = misc.split_dataset(
            environment,
            int(len(environment) * conf["holdout_fraction"]),
            misc.seed_hash(trial_seed, environment_id),
        )
        calib_dataset, test_dataset = misc.split_dataset(
            calib_test_dataset,
            int(len(calib_test_dataset) * 0.5),
            misc.seed_hash(trial_seed, environment_id),
        )
        train_datasets.append(train_dataset)
        calib_datasets.append(calib_dataset)
        test_datasets.append(test_dataset)

    if conf["imbalance_ratio"] == 1.0:
        print("No imbalance setting.")
    else:
        train_datasets, _, _ = misc.imbalance_datasets(
            conf["imbalance_ratio"],
            train_datasets,
            calib_datasets,
            test_datasets,
            seed=trial_seed,
        )

    if conf["samplesize_ratio"] != 1.0:
        train_datasets = misc.samplesize_datasets(
            conf["samplesize_ratio"],
            train_datasets,
            seed=trial_seed,
        )
        
    minority_map = build_minority_map(
        train_datasets,
        dataset_name,
        whole_dataset.num_classes,
    )

    server = Server(conf, whole_dataset, test_datasets, calib_datasets)

    clients = []
    sample_numbers = []
    for client_id in range(conf["num_client"]):
        model = server.global_models if aggregator == "fedem" else server.global_model
        clients.append(Client(conf, model, train_datasets, client_id))
        sample_numbers.append(clients[-1].sample_number)

    if conf["samplesize_ratio"] == 1.0:
        sample_weights = [1.0 for _ in sample_numbers]
    else:
        total_samples = float(sum(sample_numbers))
        sample_weights = [conf["num_client"] * count / total_samples for count in sample_numbers]

    label_count_list = None
    if conf["samplesize_ratio"] != 1.0:
        label_count_list = []
        for train_dataset in train_datasets:
            label_counts = defaultdict(int)
            for _, label in train_dataset:
                label_value = int(label.item()) if torch.is_tensor(label) else int(label)
                label_counts[label_value] += 1
            label_count_list.append(dict(label_counts))

    eval_rounds = []
    start_time = datetime.datetime.now()
    print(
        f"Aggregator: {aggregator}, Local: {conf['local_epochs']}, Global: {conf['global_epochs']}......"
    )

    for epoch in range(conf["global_epochs"]):
        print(f"=================== Global Epoch [{epoch + 1}/{conf['global_epochs']}] ===================")
        local_protos_list = []
        local_info_list = []
        featurizers = []
        classifiers = []
        models = []

        if aggregator == "fedem":
            weight_accumulators = []
            for model in server.global_models:
                weight_accumulators.append(
                    {name: torch.zeros_like(parameter) for name, parameter in model.state_dict().items()}
                )
        else:
            weight_accumulator = {
                name: torch.zeros_like(parameter)
                for name, parameter in server.global_model.state_dict().items()
            }

        for client_id, current_client in enumerate(clients):
            if aggregator in ["ours", "ours_1", "ours_12", "ours_13"]:
                diff, _, local_gaussian = current_client.local_train_gaussian(
                    server.global_model,
                    server.global_mean,
                    server.global_cov,
                    server.global_pi,
                )
                local_info_list.append(copy.deepcopy(local_gaussian))
            elif aggregator in ["pflda", "pflda_rwl"]:
                diff, _, featurizer, classifier = current_client.local_train_pflda(
                    server.global_model,
                    id_up=epoch,
                )
                featurizers.append(featurizer)
                classifiers.append(classifier)
            elif aggregator in ["ditto", "ditto_rwl"]:
                diff, _, model = current_client.local_train_ditto(server.global_model)
                models.append(model)
            elif aggregator == "fedem":
                diffs = current_client.local_train_fedem(server.global_models)
                for model_id, global_model in enumerate(server.global_models):
                    for name in global_model.state_dict():
                        weight_accumulators[model_id][name].add_(
                            sample_weights[client_id] * diffs[model_id][name]
                        )
            else:
                diff, _, local_protos = current_client.local_train_gd(
                    server.global_model,
                    server.global_protos,
                )
                local_protos_list.append(copy.deepcopy(local_protos))

            if aggregator != "fedem":
                for name in server.global_model.state_dict():
                    weight_accumulator[name].add_(sample_weights[client_id] * diff[name])

        if aggregator in ["ours", "ours_1", "ours_12", "ours_13"]:
            server.gmm_aggregation(local_info_list)
        elif aggregator in ["fedproto", "ours_3"]:
            server.proto_aggregation(local_protos_list, label_count_list)
        elif aggregator in [
            "fedavg",
            "fedprox",
            "scaffold",
            "fedrep",
            "fedwdp",
            "fedfocal",
            "fedavg_rwl",
            "fedprox_rwl",
            "scaffold_rwl",
            "fedrep_rwl",
            "fedwdp_rwl",
        ]:
            server.model_aggregate(weight_accumulator)
            excluded = ["tracked", "running_mean", "running_var"]
            if aggregator in ["fedrep", "fedrep_rwl"]:
                excluded.extend(["net.1.2", "net.1.4"])
            for current_client in clients:
                for name, parameter in server.global_model.state_dict().items():
                    if not any(keyword in name for keyword in excluded):
                        current_client.local_model.state_dict()[name].copy_(parameter.clone())
        elif aggregator in ["pflda", "pflda_rwl"]:
            coefficients = torch.tensor(sample_weights) / conf["num_client"]
            server.global_model.net[0] = misc.scalar_mul(coefficients, featurizers)
            server.global_model.net[1] = misc.scalar_mul(coefficients, classifiers)
        elif aggregator in ["ditto", "ditto_rwl"]:
            coefficients = torch.tensor(sample_weights) / conf["num_client"]
            server.global_model = misc.scalar_mul(coefficients, models)
        elif aggregator == "fedem":
            server.fedem_aggregate(weight_accumulators)
            for current_client in clients:
                for model_id, global_model in enumerate(server.global_models):
                    for name, parameter in global_model.state_dict().items():
                        if not any(
                            keyword in name
                            for keyword in ["tracked", "running_mean", "running_var"]
                        ):
                            current_client.local_models[model_id].state_dict()[name].copy_(
                                parameter.clone()
                            )

        if aggregator in ["ours", "ours_1", "ours_12", "ours_13"]:
            prediction_eval = gmm_eval(conf, clients, server, To_CSV=False, seed=trial_seed)
        elif aggregator in ["fedproto", "ours_3"]:
            prediction_eval = proto_eval(conf, clients, server, To_CSV=False, seed=trial_seed)
        elif aggregator == "fedem":
            prediction_eval = fedem_eval(conf, clients, server, To_CSV=False, seed=trial_seed)
        else:
            prediction_eval = model_eval(conf, clients, server, To_CSV=False, seed=trial_seed)

        if aggregator in ["scaffold", "scaffold_rwl"]:
            control = {
                name: torch.zeros_like(parameter.data)
                for name, parameter in server.global_model.named_parameters()
            }
            for current_client in clients:
                for name in control:
                    control[name] += current_client.local_model.delta_control[name] * (
                        sample_weights[current_client.client_id] / len(clients)
                    )
            for name in control:
                server.global_model.control[name].data += control[name].data

        prediction_eval["round"] = int(epoch)
        eval_rounds.append(prediction_eval)

    end_time = datetime.datetime.now()
    print(f"********* Aggregation Algorithm -> {aggregator}: {end_time - start_time} *********")

    eval_rounds = pd.concat(eval_rounds, axis=0, ignore_index=True)
    eval_rounds["aggregator"] = aggregator

    return eval_rounds, minority_map


def run_experiment(
    dataset_name,
    trial_seed,
    imbalance_ratio,
    experiment_type,
    heterogeneity,
    samplesize_ratio,
):
    aggregators = BASE_AGGREGATORS.copy()
    if imbalance_ratio == 0.005:
        aggregators.extend(RWL_AGGREGATORS)

    epoch_setting = EPOCHS[dataset_name][imbalance_ratio]
    tag = build_eval_tag(heterogeneity, imbalance_ratio, samplesize_ratio)
    tag_part = f"_{tag}" if tag else ""

    start_time = datetime.datetime.now()
    print(f"********* Dataset -> {dataset_name} *********")
    print(f"********* Seed -> {trial_seed} *********")

    eval_dir = os.path.join("./eval", experiment_type, dataset_name)
    os.makedirs(eval_dir, exist_ok=True)
    eval_aggregators = []
    seed_path = os.path.join(eval_dir, f"cm{tag_part}_seed{trial_seed}.pkl")
    metrics_path = default_output_path(seed_path)

    for aggregator in aggregators:
        print(f"********* Aggregation Algorithm -> {aggregator} *********")
        eval_rounds, minority_map = train_procedure(
            aggregator,
            dataset_name,
            trial_seed,
            epoch_setting["local"],
            epoch_setting["global"],
            heterogeneity=heterogeneity,
            imbalance_ratio=imbalance_ratio,
            samplesize_ratio=samplesize_ratio,
            experiment_type=experiment_type,
        )
        eval_aggregators.append(eval_rounds)

        eval_seed = pd.concat(eval_aggregators, axis=0, ignore_index=True)
        eval_seed["datasets"] = dataset_name
        eval_seed["Seed"] = trial_seed
        eval_seed["experiment type"] = experiment_type
        eval_seed["imbalance"] = imbalance_ratio if imbalance_ratio is not None else "default"
        eval_seed["heterogeneity"] = heterogeneity if heterogeneity is not None else "default"
        eval_seed["samplesize_ratio"] = (
            samplesize_ratio if samplesize_ratio is not None else "default"
        )

        metrics_result = final_metrics(eval_seed, minority_map)
        metrics_result.to_csv(metrics_path, index=False)
        final_rounds = eval_seed.groupby("aggregator")["round"].transform("max")
        eval_seed = eval_seed.loc[eval_seed["round"].eq(final_rounds)].drop(
            columns="round"
        )
        eval_seed.to_pickle(seed_path)
        current_metrics = metrics_result.loc[metrics_result["aggregator"].eq(aggregator)]
        print(f"Saved eval to: {seed_path}")
        print(f"Saved metrics to: {metrics_path}")
        print(f"Metrics for {aggregator}:")
        print(current_metrics.to_string(index=False))

    print(f"elapsed time: {datetime.datetime.now() - start_time}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--imbalance-ratio", type=float, choices=[0.005])
    parser.add_argument("--experiment-type", default="baseline")
    parser.add_argument("--heterogeneity", type=float)
    parser.add_argument("--samplesize-ratio", type=float)
    return parser.parse_args()


def main():
    args = parse_args()
    run_experiment(
        args.dataset,
        args.seed,
        args.imbalance_ratio,
        args.experiment_type,
        args.heterogeneity,
        args.samplesize_ratio,
    )


if __name__ == "__main__":
    main()
