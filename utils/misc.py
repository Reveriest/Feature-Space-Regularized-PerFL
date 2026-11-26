import copy
import hashlib
import json
import os
import sys
import random
import numpy as np
import torch
from shutil import copyfile
from collections import OrderedDict, Counter, defaultdict
from numbers import Number
import operator

def make_weights_for_balanced_classes(dataset):
    counts = Counter()
    classes = []
    for _, y in dataset:
        y = int(y)
        counts[y] += 1
        classes.append(y)

    n_classes = len(counts)

    weight_per_class = {}
    for y in counts:
        weight_per_class[y] = 1 / (counts[y] * n_classes)

    weights = torch.zeros(len(dataset))
    for i, y in enumerate(classes):
        weights[i] = weight_per_class[int(y)]

    return weights

def pdb():
    sys.stdout = sys.__stdout__
    import pdb
    print("Launching PDB, enter 'n' to step to parent function.")
    pdb.set_trace()

def seed_hash(*args):
    """
    Derive an integer hash from all args, for use as a random seed.
    """
    args_str = str(args)
    return int(hashlib.md5(args_str.encode("utf-8")).hexdigest(), 16) % (2**31)

def print_separator():
    print("="*80)

def print_row(row, colwidth=10, latex=False):
    if latex:
        sep = " & "
        end_ = "\\\\"
    else:
        sep = "  "
        end_ = ""

    def format_val(x):
        if np.issubdtype(type(x), np.floating):
            x = "{:.10f}".format(x)
        return str(x).ljust(colwidth)[:colwidth]
    print(sep.join([format_val(x) for x in row]), end_)

class _SplitDataset(torch.utils.data.Dataset):
    """Used by split_dataset"""
    def __init__(self, underlying_dataset, keys):
        super(_SplitDataset, self).__init__()
        self.underlying_dataset = underlying_dataset
        self.keys = keys
    def __getitem__(self, key):
        return self.underlying_dataset[self.keys[key]]
    def __len__(self):
        return len(self.keys)

def split_dataset(dataset, n, seed=0):
    """
    Return a pair of datasets corresponding to a random split of the given
    dataset, with n datapoints in the first dataset and the rest in the last,
    using the given random seed
    """
    assert(n <= len(dataset))
    keys = list(range(len(dataset)))
    np.random.RandomState(seed).shuffle(keys)
    keys_1 = keys[:n]
    keys_2 = keys[n:]
    return _SplitDataset(dataset, keys_1), _SplitDataset(dataset, keys_2)

def random_pairs_of_minibatches(minibatches):
    perm = torch.randperm(len(minibatches)).tolist()
    pairs = []

    for i in range(len(minibatches)):
        j = i + 1 if i < (len(minibatches) - 1) else 0

        xi, yi = minibatches[perm[i]][0], minibatches[perm[i]][1]
        xj, yj = minibatches[perm[j]][0], minibatches[perm[j]][1]

        min_n = min(len(xi), len(xj))

        pairs.append(((xi[:min_n], yi[:min_n]), (xj[:min_n], yj[:min_n])))

    return pairs

def meanseloss(network, loader, weights, device,envidx=0):
    correct = 0
    total = 0
    weights_offset = 0

    network.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            p = network.predict(x, envidx)
            if weights is None:
                batch_weights = torch.ones(len(x))
            else:
                batch_weights = weights[weights_offset : weights_offset + len(x)]
                weights_offset += len(x)
            batch_weights = batch_weights.to(device)
            if p.size(1) == 1:
                correct += torch.sqrt((((p-y)**2).sum())* batch_weights.sum()).item()
            else:
                correct += torch.sqrt((((p-y)**2).sum())* batch_weights.sum()).item()
            
            total += batch_weights.sum().item()
    network.train()
    return correct / total
    
def accuracy(network, loader, weights, device,envidx=0):
    correct = 0
    total = 0
    weights_offset = 0

    network.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            p = network.predict(x, envidx)
            if weights is None:
                batch_weights = torch.ones(len(x))
            else:
                batch_weights = weights[weights_offset : weights_offset + len(x)]
                weights_offset += len(x)
            batch_weights = batch_weights.to(device)
            if p.size(1) == 1:
                correct += (p.gt(0).eq(y).float() * batch_weights.view(-1, 1)).sum().item()
            else:
                correct += (p.argmax(1).eq(y).float() * batch_weights).sum().item()
            total += batch_weights.sum().item()
    network.train()

    return correct / total

class Tee:
    def __init__(self, fname, mode="a"):
        self.stdout = sys.stdout
        self.file = open(fname, mode)

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()

class ParamDict(OrderedDict):
    """Code adapted from https://github.com/Alok/rl_implementations/tree/master/reptile.
    A dictionary where the values are Tensors, meant to represent weights of
    a model. This subclass lets you perform arithmetic on weights directly."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, *kwargs)

    def _prototype(self, other, op):
        if isinstance(other, Number):
            return ParamDict({k: op(v, other) for k, v in self.items()})
        elif isinstance(other, dict):
            return ParamDict({k: op(self[k], other[k]) for k in self})
        else:
            raise NotImplementedError

    def __add__(self, other):
        return self._prototype(other, operator.add)

    def __rmul__(self, other):
        return self._prototype(other, operator.mul)

    __mul__ = __rmul__

    def __neg__(self):
        return ParamDict({k: -v for k, v in self.items()})

    def __rsub__(self, other):
        # a- b := a + (-b)
        return self.__add__(other.__neg__())

    __sub__ = __rsub__

    def __truediv__(self, other):
        return self._prototype(other, operator.truediv)


def scalar_mul(scalar_list, net_list, rg=False):
    result = copy.deepcopy(net_list[0])
    with torch.no_grad():
        result_dict = result.state_dict()
        nets_dict = [neti.state_dict() for neti in net_list]
        weight_keys = list(result_dict.keys())
        for key in weight_keys:
            result_dict[key] *= 0
            for j in range(len(scalar_list)):
                result_dict[key] = result_dict[key] + scalar_list[j]*nets_dict[j][key]
        result.load_state_dict(result_dict)
        return result

def imbalance_datasets(imbalance_ratio, train_datasets, calib_datasets, test_datasets, seed=1):
    random.seed(seed)
    np.random.seed(seed)
    
    def process_dataset_with_ratios(dataset, keep_ratios):
        labels = []
        for i in range(len(dataset)):
            _, label = dataset[i]
            labels.append(label.item() if torch.is_tensor(label) else label)
        
        kept_indices = []
        for class_id in sorted(keep_ratios.keys()):
            class_indices = [i for i, label in enumerate(labels) if label == class_id]
            if len(class_indices) == 0:
                continue
                
            original_count = len(class_indices)
            keep_count = int(original_count * keep_ratios[class_id])
            keep_count = max(1, keep_count)
            
            if keep_count >= original_count:
                kept_class_indices = class_indices
            else:
                # Randomly select indices to keep
                # kept_class_indices = random.sample(class_indices, keep_count)
                
                # Randomly select start point for contiguous selection
                max_start = original_count - keep_count
                start_idx = random.randint(0, max_start)
                kept_class_indices = class_indices[start_idx:start_idx + keep_count]
            kept_indices.extend(kept_class_indices)

        subset_dataset = torch.utils.data.Subset(dataset, kept_indices)
        return subset_dataset
    
    def get_class_distribution(dataset):
        labels = []
        for i in range(len(dataset)):
            _, label = dataset[i]
            labels.append(label.item() if torch.is_tensor(label) else label)
        return Counter(labels)
    
    imbalanced_train_datasets = []
    imbalanced_calib_datasets = []
    imbalanced_test_datasets = []
    for client_id in range(len(train_datasets)):
        print(f"Imbalancing dataset of client {client_id}...")
        
        train_class_counts = get_class_distribution(train_datasets[client_id])
        classes = list(train_class_counts.keys())
        C = len(classes)
        
        if C <= 1:
            imbalanced_train_datasets.append(train_datasets[client_id])
            imbalanced_calib_datasets.append(calib_datasets[client_id])
            imbalanced_test_datasets.append(test_datasets[client_id])
            continue
        
        permuted_classes = classes.copy()
        random.shuffle(permuted_classes)
        
        step = (1 - imbalance_ratio) / (C - 1) if C > 1 else 0
        
        keep_ratios = {}
        for i, class_id in enumerate(permuted_classes):
            if i == C - 1:
                keep_ratios[class_id] = 1.0
            else:
                ratio = imbalance_ratio + i * step
                keep_ratios[class_id] = ratio
        print(f"  Class permutation order: {permuted_classes}")
        print(f"  Keep ratios: {keep_ratios}")
        
        imbalanced_train = process_dataset_with_ratios(train_datasets[client_id], keep_ratios)
        imbalanced_train_datasets.append(imbalanced_train)
        
        imbalanced_calib = process_dataset_with_ratios(calib_datasets[client_id], keep_ratios)
        imbalanced_calib_datasets.append(imbalanced_calib)
        
        imbalanced_test = process_dataset_with_ratios(test_datasets[client_id], keep_ratios)
        imbalanced_test_datasets.append(imbalanced_test)
        
    return imbalanced_train_datasets, imbalanced_calib_datasets, imbalanced_test_datasets

def samplesize_datasets(samplesize_ratio, train_datasets, seed=1):

    if samplesize_ratio <= 0 or samplesize_ratio > 1:
        raise ValueError("samplesize_ratio should be in (0, 1].")

    num_clients = len(train_datasets)
    if num_clients <= 1 or samplesize_ratio == 1.0:
        return train_datasets

    rng = np.random.RandomState(seed)
    client_order = list(range(num_clients))
    rng.shuffle(client_order)

    decay = samplesize_ratio ** (1.0 / (num_clients - 1))
    base_count = len(train_datasets[client_order[0]])

    keep_ratios = {}
    target_counts = {}
    for rank, client_id in enumerate(client_order):
        ratio = decay ** rank
        target_count = int(round(base_count * ratio))
        target_count = max(1, min(len(train_datasets[client_id]), target_count))
        keep_ratios[client_id] = ratio
        target_counts[client_id] = target_count

    def get_label(y):
        return int(y.item()) if torch.is_tensor(y) else int(y)

    def stratified_subset(dataset, target_count, client_id):
        if target_count >= len(dataset):
            return dataset

        class_indices = defaultdict(list)
        for idx in range(len(dataset)):
            _, label = dataset[idx]
            class_indices[get_label(label)].append(idx)

        raw_counts = {}
        floors = {}
        fractions = {}
        for label, indices in class_indices.items():
            expected = len(indices) * target_count / len(dataset)
            floor_count = min(len(indices), int(np.floor(expected)))
            raw_counts[label] = expected
            floors[label] = floor_count
            fractions[label] = expected - floor_count

        selected_counts = floors.copy()
        remaining = target_count - sum(selected_counts.values())
        labels_by_fraction = sorted(
            class_indices.keys(),
            key=lambda label: (fractions[label], raw_counts[label]),
            reverse=True,
        )

        while remaining > 0:
            updated = False
            for label in labels_by_fraction:
                if selected_counts[label] < len(class_indices[label]):
                    selected_counts[label] += 1
                    remaining -= 1
                    updated = True
                    if remaining == 0:
                        break
            if not updated:
                break

        local_rng = np.random.RandomState(seed + client_id)
        kept_indices = []
        for label, indices in class_indices.items():
            indices = indices.copy()
            local_rng.shuffle(indices)
            kept_indices.extend(indices[:selected_counts[label]])
        local_rng.shuffle(kept_indices)

        return torch.utils.data.Subset(dataset, kept_indices)

    samplesized_train_datasets = []
    for client_id, dataset in enumerate(train_datasets):
        print(f"Sample-size imbalancing dataset of client {client_id}...")
        print(
            f"  Rank: {client_order.index(client_id)}, "
            f"keep ratio: {keep_ratios[client_id]:.6f}, "
            f"target/current: {target_counts[client_id]}/{len(dataset)}"
        )
        samplesized_train_datasets.append(
            stratified_subset(dataset, target_counts[client_id], client_id)
        )

    print(f"  Client order from largest to smallest: {client_order}")
    return samplesized_train_datasets