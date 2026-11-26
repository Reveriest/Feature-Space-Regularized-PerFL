import os
import copy
import numpy as np
from collections import defaultdict
import torch
from torch.utils.data import DataLoader

import network.networks as networks


class Server(object):
    
    def __init__(self, conf, WholeDataset, eval_datasets, calib_datasets):
        self.conf = conf
        self.global_model = networks.WholeFish(
            WholeDataset.input_shape,
            WholeDataset.num_classes,
            self.conf,
        ).to('cuda')
        
        if self.conf.get("aggregator") == "fedem":
            self.M = self.conf.get("M", 3)
            self.global_models = [
                networks.WholeFish(WholeDataset.input_shape,
                                   WholeDataset.num_classes,
                                   self.conf).to('cuda') 
                for _ in range(self.M)
            ]        
        self.global_protos = None
        self.global_mean = None
        self.global_cov = None
        self.global_pi = None
        self.eval_loader_list = [DataLoader(eval_datasets[i],
                                            batch_size=32,
                                            shuffle=False, drop_last=False)
                                             for i in range(len(eval_datasets))]
        self.calib_loader_list = [DataLoader(calib_datasets[i],
                                            batch_size=32,
                                            shuffle=False, drop_last=False)
                                             for i in range(len(calib_datasets))]
        
        for k, v in self.global_model.named_parameters():
            self.global_model.control[k] = torch.zeros_like(v.data)
            self.global_model.delta_control[k] = torch.zeros_like(v.data)
            self.global_model.delta_y[k] = torch.zeros_like(v.data)
        
    def model_aggregate(self, weight_accumulator):
        for name, data in self.global_model.state_dict().items():
            if not any(keyword in name for keyword in ["tracked", "running_mean", "running_var"]):
                data = data.to(torch.float32)
                update_per_layer = weight_accumulator[name] * 1/self.conf["num_client"]
                if data.type() != update_per_layer.type():
                    data.add_(update_per_layer.to(torch.float32))
                else:
                    data.add_(update_per_layer)

    def fedem_aggregate(self, weight_accumulators):
        for m in range(self.M):
            for name, data in self.global_models[m].state_dict().items():
                if not any(keyword in name for keyword in ["tracked", "running_mean", "running_var"]):
                    data = data.to(torch.float32)
                    update_per_layer = weight_accumulators[m][name] * (1.0 / self.conf["num_client"])
                    if data.type() != update_per_layer.type():
                        data.add_(update_per_layer.to(torch.float32))
                    else:
                        data.add_(update_per_layer)
    
    def proto_aggregation(self, local_protos_list, label_count_list=None):
        agg_protos_label = defaultdict(list)
        use_weighted = label_count_list is not None
        if use_weighted and len(label_count_list) != len(local_protos_list):
            raise ValueError("Length mismatch.")
        for c_id, local_protos in enumerate(local_protos_list):
            for label in local_protos.keys():
                if use_weighted:
                    count_dict = label_count_list[c_id]
                    n_label = float(count_dict.get(label, 0))
                    agg_protos_label[label].append((local_protos[label], n_label))
                else:
                    agg_protos_label[label].append(local_protos[label])
        for [label, proto_list] in agg_protos_label.items():
            if not use_weighted:
                if len(proto_list) > 1:
                    proto = 0 * proto_list[0].data
                    for i in proto_list:
                        proto += i.data
                    agg_protos_label[label] = proto / len(proto_list)
                else:
                    agg_protos_label[label] = proto_list[0].data
            else:
                protos = [item[0] for item in proto_list]
                counts = torch.tensor(
                    [max(item[1], 0.0) for item in proto_list],
                    dtype=torch.float32,
                    device=protos[0].device,
                )
                if torch.sum(counts) <= 0:
                    weights = torch.ones_like(counts) / len(counts)
                else:
                    weights = counts / torch.sum(counts)
                proto = torch.zeros_like(protos[0].data)
                for i, p in enumerate(protos):
                    proto += p.data * weights[i]
                agg_protos_label[label] = proto
        self.global_protos = agg_protos_label
        
    def gmm_aggregation(self, local_info_list):
        all_classes = set()
        for local_info in local_info_list:
            all_classes.update(local_info["mean"].keys())
            
        g_mean = defaultdict(list)
        g_cov = defaultdict(list)
        g_pi = defaultdict(list)
        for class_id in all_classes:
            means = []
            covs = []
            nums = []
            for local_info in local_info_list:
                if class_id in local_info["mean"]:
                    means.append(local_info["mean"][class_id])
                    covs.append(local_info["cov"][class_id])
                    nums.append(local_info["num"][class_id])
            means = [m.detach() for m in means]
            covs = [c.detach() for c in covs]
            g_mean[class_id] = torch.stack(means)
            g_cov[class_id] = torch.stack(covs)
            total_samples = sum(nums)
            if total_samples > 0:
                g_pi[class_id] = torch.tensor(nums) / total_samples
            else:
                g_pi[class_id] = torch.zeros(len(nums))
        self.global_mean = g_mean
        self.global_cov = g_cov
        self.global_pi = g_pi
        

    def save_gmm_parameters(self, dirspath=None, seed=None):
        mean_dict = {k: v.cpu().detach().numpy() for k, v in self.global_mean.items()}
        cov_dict = {k: v.cpu().detach().numpy() for k, v in self.global_cov.items()}
        pi_dict = {k: v.cpu().detach().numpy() for k, v in self.global_pi.items()}
        
        if dirspath is not None:
            if not os.path.exists(dirspath):
                os.makedirs(dirspath, exist_ok=True)
            savepath = os.path.join(dirspath, f"gmm.npz")
        elif seed is not None:
            if not os.path.exists(f"./weights/{self.conf['datasets']}/{self.conf['aggregator']}_seed{seed}"):
                os.makedirs(f"./weights/{self.conf['datasets']}/{self.conf['aggregator']}_seed{seed}", exist_ok=True)
            savepath = f"./weights/{self.conf['datasets']}/{self.conf['aggregator']}_seed{seed}/gmm.npz"
        else:
            if not os.path.exists(f"./weights/{self.conf['datasets']}/{self.conf['aggregator']}"):
                os.makedirs(f"./weights/{self.conf['datasets']}/{self.conf['aggregator']}", exist_ok=True)
            savepath = f"./weights/{self.conf['datasets']}/{self.conf['aggregator']}/gmm.npz"
        np.savez(
            savepath,
            mean_dict=mean_dict,
            cov_dict=cov_dict,
            pi_dict=pi_dict
        )
        print(f"Global GMM parameters saved to {savepath}......")
        
    def load_gmm_parameters(self, dirspath=None, seed=None):
        if dirspath is not None:
            loadpath = os.path.join(dirspath, f"gmm.npz")
        elif seed is not None:
            loadpath = f"./weights/{self.conf['datasets']}/{self.conf['aggregator']}_seed{seed}/gmm.npz"
        else:
            loadpath = f"./weights/{self.conf['datasets']}/{self.conf['aggregator']}/gmm.npz"
        param = np.load(loadpath, allow_pickle=True)
        mean_dict = {int(k): torch.tensor(v) for k, v in param['mean_dict'].item().items()}
        cov_dict = {int(k): torch.tensor(v) for k, v in param['cov_dict'].item().items()}
        pi_dict = {int(k): torch.tensor(v) for k, v in param['pi_dict'].item().items()}
        mean_default = defaultdict(list)
        cov_default = defaultdict(list)
        pi_default = defaultdict(list)
        mean_default.update(mean_dict)
        cov_default.update(cov_dict)
        pi_default.update(pi_dict)
        self.global_mean = mean_default
        self.global_cov = cov_default
        self.global_pi = pi_default
        print(f"Global GMM parameters loaded from {loadpath}......")