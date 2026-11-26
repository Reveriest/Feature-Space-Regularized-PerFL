import os
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score
from collections import defaultdict

from utils import misc
from data.fast_data_loader import InfiniteDataLoader
from utils.utils_matrix import *

class Client(object):
    
    def __init__(self, conf, model, train_datasets, id=-1):
        self.conf = conf
        if self.conf.get("aggregator") == "fedem":
            self.M = self.conf.get("M", 3)
            self.local_models = [copy.deepcopy(model[m]) for m in range(self.M)]
            self.pi = (torch.ones(self.M) / self.M).to('cuda')
        else:
            self.local_model = copy.deepcopy(model)
        self.client_id = id
        self.train_dataset = train_datasets[id]
        self.train_loader = InfiniteDataLoader(
                                dataset=self.train_dataset,
                                weights=None,
                                batch_size=self.conf["batch_size"],
                                # batch_size=len(self.train_dataset),
                                num_workers=0,
                                index=self.client_id,
                            )
        self.sample_number = len(self.train_dataset)
        self.loss_mse = nn.MSELoss()
        self.loss_con = ContrastiveDistributionLoss()
        self.loss_orth = OrthonormalPrototypeLoss()
        self.lr_decay = 0.99
        
        self.label_counts = defaultdict(int)
        for _, y in self.train_dataset:
            y_int = int(y.item()) if torch.is_tensor(y) else int(y)
            self.label_counts[y_int] += 1
        self.inverse_ratio = torch.zeros(self.conf.get("no_classes"))
        for c in range(self.conf.get("no_classes")):
            count = self.label_counts.get(c, 0)
            if count > 0:
                self.inverse_ratio[c] = 1.0 / count
        sum_ratio = self.inverse_ratio.sum()
        if sum_ratio > 0:
            self.inverse_ratio = self.inverse_ratio / sum_ratio

    def local_train_gd(self, model, global_protos=None):
        device = next(model.parameters()).device
        self.local_model.to(device)
        # Global model supplants local model（FedAvg/FedProx/SCAFFOLD/FedWDP/FedFocal）
        if self.conf["aggregator"] in ["fedavg", "fedprox", "scaffold", "fedwdp", "fedfocal",
                                       "fedavg_rwl", "fedprox_rwl", "scaffold_rwl", "fedwdp_rwl"]:
            for name, param in model.state_dict().items():
                if not any(keyword in name for keyword in ["tracked", "running_mean", "running_var"]):
                    self.local_model.state_dict()[name].copy_(param.clone())
        elif self.conf["aggregator"] in ["fedrep", "fedrep_rwl"]:
            for name, param in model.state_dict().items():
                if not any(keyword in name for keyword in ["tracked", "running_mean", "running_var", "net.1.2", "net.1.4"]):
                    self.local_model.state_dict()[name].copy_(param.clone())
        else:
            pass
        if self.conf.get("aggregator") == "fedfocal":
            alpha_weights = self.inverse_ratio.to(device)
            criterion = FocalLoss(alpha=alpha_weights, gamma=2.0, reduction='mean')
        elif self.conf.get("aggregator").endswith("_rwl"):
            weight_tensor = self.inverse_ratio.to(device)
            criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        else:
            criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.local_model.parameters(),
                                     lr=self.conf["lr"],
                                     weight_decay=0.0)
        self.local_model.train()
        protos = defaultdict(list)
        for e in range(self.conf["local_epochs"]):
            labels, outputs, rep = [], [], []
            for inputs_b, labels_b in self.train_loader:
                inputs_b = inputs_b.to(device)
                labels_b = labels_b.to(device)
                outputs_b, rep_b = self.local_model(inputs_b)
                labels.append(labels_b)
                outputs.append(outputs_b)
                rep.append(rep_b)
            labels = torch.cat(labels, dim=0)
            outputs = torch.cat(outputs, dim=0)
            rep = torch.cat(rep, dim=0)
            loss = criterion(outputs, labels)
            if self.conf["aggregator"] in ["fedproto", "ours_3"]:
                if global_protos is not None:
                    proto_new = copy.deepcopy(rep.detach())
                    for i, label in enumerate(labels):
                        if type(global_protos[label.item()]) != type([]):
                            proto_new[i, :] = global_protos[label.item()].data
                    loss += 0.01 * self.loss_mse(proto_new, rep)
                    if self.conf["aggregator"] == "ours_3":
                        loss += 0.001*self.conf["lambda"] * self.loss_con(rep, proto_new, labels)
                        loss += 0.001*self.conf["lambda"] * self.loss_orth(rep, labels)
            if e == self.conf["local_epochs"] - 1:    
                for i, label in enumerate(labels):
                    protos[label.item()].append(rep[i, :].detach().data)
            # FedWDP
            if self.conf.get("aggregator") in ["fedwdp", "fedwdp_rwl"]:
                swd_loss = 0.0
                global_modules = list(model.modules())
                local_modules = list(self.local_model.modules())
                for mod_l, mod_g in zip(local_modules, global_modules):
                    if isinstance(mod_l, (nn.Conv2d, nn.Linear)):
                        param_l = mod_l.weight
                        param_g = mod_g.weight.clone().detach()
                        w_l = param_l.view(param_l.shape[0], -1)
                        w_g = param_g.view(param_g.shape[0], -1)
                        swd_loss += sliced_wasserstein_distance(w_l, w_g,
                                                                num_projections=50,
                                                                device=device)
                swd_mu = self.conf.get("lambda1", 0.01)
                loss += swd_mu * swd_loss
            optimizer.zero_grad()
            loss.backward()
            if self.conf.get("aggregator") in ["fedwdp", "fedwdp_rwl"]:
                torch.nn.utils.clip_grad_norm_(self.local_model.parameters(),
                                               max_norm=0.1, norm_type=2.0)
            optimizer.step()                
            
            if self.conf["aggregator"] in ["fedavg", "fedavg_rwl"]:
                pass
            elif self.conf["aggregator"] in ["fedprox", "fedprox_rwl"]:
                with torch.no_grad():
                    Conv_Weight_List, Linear_Weight_List = [], []
                    for Layer in model.modules():
                        if isinstance(Layer, nn.Conv2d):
                            Conv_Weight_List.append(Layer.weight.clone().detach())
                        elif isinstance(Layer, nn.Linear):
                            Linear_Weight_List.append(Layer.weight.clone().detach())
                    idx_Conv, idx_Linear = 0, 0
                    for Layer in self.local_model.modules():
                        if isinstance(Layer, nn.Conv2d):
                            Layer.weight.add_(self.conf['lr'] * (Layer.weight.data - Conv_Weight_List[idx_Conv]))
                            idx_Conv += 1
                        elif isinstance(Layer, nn.Linear):
                            Layer.weight.add_(self.conf['lr'] * (Layer.weight.data - Linear_Weight_List[idx_Linear]))
                            idx_Linear += 1
            elif self.conf["aggregator"] in ["scaffold", "scaffold_rwl"]:
                with torch.no_grad():
                    for k, v in self.local_model.named_parameters():
                        v.data = v.data + self.conf['lr'] * (self.local_model.control[k] - model.control[k])
            elif self.conf["aggregator"] in ["fedrep", "fedproto", "ours_3", "local", "fedfocal", "fedwdp",
                                             "fedrep_rwl", "local_rwl", "fedwdp_rwl"]:
                pass
            if e%10 == 0:
                print(f'Client {self.client_id}, Epoch [{e}/{self.conf["local_epochs"]}], loss: {loss.item()}')
        if self.conf["aggregator"] in ["scaffold", "scaffold_rwl"]:
            temp = {}
            for k, v in self.local_model.named_parameters():
                temp[k] = v.data.clone()
            for k, v in model.named_parameters():
                self.local_model.control[k] = (self.local_model.control[k] - model.control[k] 
                                + (v.data - temp[k]) 
                                / (self.conf['local_epochs'] * self.conf['lr'] * len(self.train_loader)))
                self.local_model.delta_y[k] = temp[k] - v.data
                self.local_model.delta_control[k] = self.local_model.control[k] - model.control[k]
        para = dict()
        diff = dict()
        for name, data in self.local_model.state_dict().items():
            para[name] = copy.deepcopy(data)
            diff[name] = (data - model.state_dict()[name])
        return diff, para, agg_func(protos)


    # ------------------ Local Gaussian & Global GMM ------------------
    def local_train_gaussian(self, model, global_mean=None, global_cov=None, global_pi=None):
        device = next(model.parameters()).device
        self.local_model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.local_model.parameters(),
                                     lr=self.conf["lr"],
                                     weight_decay=0.1*self.conf["lr"])
        self.local_model.train()
        for e in range(self.conf["local_epochs"]):
            labels, outputs, rep = [], [], []
            for inputs_b, labels_b in self.train_loader:
                inputs_b, labels_b = inputs_b.to(device), labels_b.to(device)
                outputs_b, rep_b = self.local_model(inputs_b)
                labels.append(labels_b)
                outputs.append(outputs_b)
                rep.append(rep_b)
            labels = torch.cat(labels, dim=0)
            outputs = torch.cat(outputs, dim=0)
            rep = torch.cat(rep, dim=0)
            loss = criterion(outputs, labels)
            if global_mean is not None:
                proto_new = copy.deepcopy(rep.detach())
                Identity_G = torch.ones((self.conf["no_classes"],
                                        rep.shape[1], rep.shape[1]),
                                        device=device).detach()
                unique_labels = torch.unique(labels.detach())
                indices_labels = [torch.where(labels == l)[0] for l in unique_labels]
                for i, l in enumerate(unique_labels):
                    indices = indices_labels[i]
                    features_l = rep[indices]
                    mu_gmm_l = global_mean[l.item()].detach().to(device)
                    cov_gmm_l = global_cov[l.item()].detach().to(device)
                    pi_gmm_l = global_pi[l.item()].detach().to(device)
                    proto_l = torch.sum(mu_gmm_l * (pi_gmm_l / pi_gmm_l.sum()).unsqueeze(1), dim=0).detach()
                    proto_new[indices, :] = proto_l.data
                    mu_l = torch.mean(features_l, dim=0)
                    cov_l = features_l.t().cov(correction=1)

                    loss += self.conf["lambda1"] * wd_gmm_per_class(mu_l, cov_l, mu_gmm_l, cov_gmm_l, pi_gmm_l)
                    if self.conf["aggregator"] in ["ours", "ours_12"]:
                        cov_offdiag = cov_l - torch.diag(torch.diag(cov_l))
                        loss += self.conf["lambda2"] * torch.sum(torch.abs(cov_offdiag))
                if self.conf["aggregator"] in ["ours", "ours_13"]:
                    loss += self.conf["lambda3"] * (self.loss_con(rep, proto_new, labels, Identity_G) + self.loss_orth(rep, labels))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if e%10 == 0:
                print(f'Client {self.client_id}, Epoch [{e}/{self.conf["local_epochs"]}], loss: {loss.item()}')
                
            if e == self.conf["local_epochs"] - 1:
                with torch.no_grad():
                    classes_mean, classes_cov, classes_num = defaultdict(list), defaultdict(list), defaultdict(list)
                    unique_labels = torch.unique(labels.detach())
                    indices_labels = [torch.where(labels == l)[0] for l in unique_labels]
                    for i, l in enumerate(unique_labels):
                        indices = indices_labels[i]
                        features_class = rep.detach()[indices]
                        mean_class = torch.mean(features_class, dim=0)
                        cov_class = features_class.t().cov(correction=1)
                        classes_mean[l.item()] = mean_class.detach()
                        classes_cov[l.item()] = cov_class.detach()
                        classes_num[l.item()] = len(indices)
                    local_gaussian = {
                        "mean": classes_mean,
                        "cov": classes_cov,
                        "num": classes_num,
                    }
        para = dict()
        diff = dict()
        for name, data in self.local_model.state_dict().items():
            para[name] = copy.deepcopy(data)
            diff[name] = (data - model.state_dict()[name])
        return diff, para, local_gaussian

    # ------------------ w/ Diminishing learning Rate ------------------
    def local_train_gaussian_decay(self, model, global_mean=None, global_cov=None, global_pi=None,
                                   global_epoch=0):
        device = next(model.parameters()).device
        self.local_model.to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.local_model.parameters(),
                                     lr=self.conf["lr"]/((global_epoch+1)**0.5),
                                     weight_decay=self.conf["weight_decay"])
        self.local_model.train()
        for e in range(self.conf["local_epochs"]):
            labels, outputs, rep = [], [], []
            for inputs_b, labels_b in self.train_loader:
                inputs_b, labels_b = inputs_b.to(device), labels_b.to(device)
                outputs_b, rep_b = self.local_model(inputs_b)
                labels.append(labels_b)
                outputs.append(outputs_b)
                rep.append(rep_b)
            labels = torch.cat(labels, dim=0)
            outputs = torch.cat(outputs, dim=0)
            rep = torch.cat(rep, dim=0)
            loss = criterion(outputs, labels)
            if global_mean is not None:
                proto_new = copy.deepcopy(rep.detach())
                Identity_G = torch.ones((self.conf["no_classes"],
                                        rep.shape[1], rep.shape[1]),
                                        device=device).detach()     
                unique_labels = torch.unique(labels.detach())
                indices_labels = [torch.where(labels == l)[0] for l in unique_labels]
                for i, l in enumerate(unique_labels):
                    indices = indices_labels[i]
                    features_l = rep[indices]
                    mu_gmm_l = global_mean[l.item()].detach().to(device)
                    cov_gmm_l = global_cov[l.item()].detach().to(device)
                    pi_gmm_l = global_pi[l.item()].detach().to(device)
                    proto_l = torch.sum(mu_gmm_l * (pi_gmm_l / pi_gmm_l.sum()).unsqueeze(1), dim=0).detach()
                    proto_new[indices, :] = proto_l.data
                    mu_l = torch.mean(features_l, dim=0)
                    cov_l = features_l.t().cov(correction=1)
                    loss += self.conf["lambda1"] * wd_gmm_per_class(mu_l, cov_l, mu_gmm_l, cov_gmm_l, pi_gmm_l)
                    if self.conf["aggregator"] in ["ours", "ours_12"]:
                        cov_offdiag = cov_l - torch.diag(torch.diag(cov_l))
                        loss += self.conf["lambda2"] * torch.sum(torch.abs(cov_offdiag))
                if self.conf["aggregator"] in ["ours", "ours_13"]:
                    loss += self.conf["lambda3"] * (self.loss_con(rep, proto_new, labels, Identity_G) + self.loss_orth(rep, labels))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if e%10 == 0:
                print(f'Client {self.client_id}, Epoch [{e}/{self.conf["local_epochs"]}], loss: {loss.item()}')
                
            if e == self.conf["local_epochs"] - 1:
                with torch.no_grad():
                    classes_mean, classes_cov, classes_num = defaultdict(list), defaultdict(list), defaultdict(list)
                    unique_labels = torch.unique(labels.detach())
                    indices_labels = [torch.where(labels == l)[0] for l in unique_labels]
                    for i, l in enumerate(unique_labels):
                        indices = indices_labels[i]
                        features_class = rep.detach()[indices]
                        mean_class = torch.mean(features_class, dim=0)
                        cov_class = features_class.t().cov(correction=1)
                        classes_mean[l.item()] = mean_class.detach()
                        classes_cov[l.item()] = cov_class.detach()
                        classes_num[l.item()] = len(indices)
                    local_gaussian = {
                        "mean": classes_mean,
                        "cov": classes_cov,
                        "num": classes_num,
                    }
        para = dict()
        diff = dict()
        for name, data in self.local_model.state_dict().items():
            para[name] = copy.deepcopy(data)
            diff[name] = (data - model.state_dict()[name])
        return diff, para, local_gaussian

    # ------------------ Ditto ------------------
    def local_train_ditto(self, model):
        device = next(model.parameters()).device
        self.local_model.to(device)
        weight_tensor = self.inverse_ratio.to(device) if self.conf.get("aggregator") == "ditto_rwl" else None
        model_i = copy.deepcopy(model)
        optimizer_g = torch.optim.Adam(
            model_i.parameters(),
            lr=self.conf["lr"],
        )
        optimizer_l = torch.optim.Adam(
            self.local_model.parameters(),
            lr=self.conf["lr"],
        )
        model_i.train()
        self.local_model.train()
        print(f"    Train the centralized global model ---")
        for e in range(self.conf["local_epochs"]):
            labels, outputs = [], []
            for inputs_b, labels_b in self.train_loader:
                inputs_b, labels_b = inputs_b.to(device), labels_b.to(device)
                outputs_b, _ = model_i(inputs_b)
                labels.append(labels_b)
                outputs.append(outputs_b)
            labels = torch.cat(labels, dim=0)
            outputs = torch.cat(outputs, dim=0)
            loss = nn.functional.cross_entropy(outputs, labels, weight=weight_tensor)
            optimizer_g.zero_grad()
            loss.backward()
            optimizer_g.step()
            if e%10 == 0:
                print(f'Central Client {self.client_id}, Epoch [{e}/{self.conf["local_epochs"]}], loss: {loss.item()}')
                
        print(f"    Train localized models ---")
        for e in range(self.conf["local_epochs"]):
            targets, Yi_l = [], []
            for xi, yi in self.train_loader:
                xi, yi = xi.to(device), yi.to(device)
                yi_l, _ = self.local_model(xi)
                targets.append(yi)
                Yi_l.append(yi_l)
            targets = torch.cat(targets, dim=0)
            Yi_l = torch.cat(Yi_l, dim=0)
            loss_l = nn.functional.cross_entropy(Yi_l, targets, weight=weight_tensor)
            optimizer_l.zero_grad()
            loss_l.backward()
            optimizer_l.step()
            beta3 = 0.99
            self.local_model = misc.scalar_mul([beta3, 1-beta3], [self.local_model, model])
            if e%10 == 0:
                print(f'Local Client {self.client_id}, Epoch [{e}/{self.conf["local_epochs"]}], loss: {loss_l.item()}')        
        para = dict()
        diff = dict()
        for name, data in self.local_model.state_dict().items():
            para[name] = copy.deepcopy(data)
            diff[name] = (data - model.state_dict()[name])
        return diff, para, model_i
              
    # ------------------ PFL-DA ------------------
    def local_train_pflda(self, model, id_up=0):
        device = next(model.parameters()).device
        self.local_model.to(device)
        weight_tensor = self.inverse_ratio.to(device) if self.conf.get("aggregator") == "pflda_rwl" else None
        featurizer_i = copy.deepcopy(model.net[0])
        classifier_i = copy.deepcopy(model.net[1])
        optimizer_g = torch.optim.Adam(
                    list(classifier_i.parameters())+list(featurizer_i.parameters())+list(self.local_model.net[0].parameters()),
                    lr=self.conf["lr"]*self.lr_decay**(1+id_up//10),
                    weight_decay=self.conf['weight_decay'])
        optimizer_l = torch.optim.Adam(
                self.local_model.net[1].parameters(),
                lr=self.conf["lr"]*self.lr_decay**(1+id_up//10),
                weight_decay=self.conf['weight_decay'])
        self.local_model.train()
        featurizer_i.train()
        classifier_i.train()
        for e in range(self.conf["local_epochs"]):
            inputs, labels, outputs = [], [], []
            for inputs_b, labels_b in self.train_loader:
                inputs_b, labels_b = inputs_b.to(device), labels_b.to(device)
                rep_b = self.local_model.net[0](inputs_b)
                outputs_b = classifier_i(rep_b)
                inputs.append(inputs_b)
                labels.append(labels_b)
                outputs.append(outputs_b)
            inputs = torch.cat(inputs, dim=0)
            labels = torch.cat(labels, dim=0)
            outputs = torch.cat(outputs, dim=0)
            loss = nn.functional.cross_entropy(outputs, labels, weight=weight_tensor)
            optimizer_g.zero_grad()
            optimizer_l.zero_grad()
            loss.backward()
            optimizer_g.step()
            Yi, Yi_L = [], []
            for xi, yi in self.train_loader:
                xi, yi = xi.to(device), yi.to(device)
                with torch.no_grad():
                    featuresi = self.local_model.net[0](xi)
                yi_l = self.local_model.net[1](featuresi)
                Yi.append(yi)
                Yi_L.append(yi_l)
            Yi = torch.cat(Yi, dim=0)
            Yi_L = torch.cat(Yi_L, dim=0)
            loss1 = nn.functional.cross_entropy(Yi_L, Yi, weight=weight_tensor)
            loss1.backward()
            optimizer_l.step()
            beta3 = 0.99
            self.local_model.net[1] = misc.scalar_mul([beta3, 1-beta3], [model.net[1], self.local_model.net[1]])
            if e%10 == 0:
                print(f'Client {self.client_id}, Epoch [{e}/{self.conf["local_epochs"]}], loss: {loss.item()}')
        para = dict()
        diff = dict()
        for name, data in self.local_model.state_dict().items():
            para[name] = copy.deepcopy(data)
            diff[name] = (data - model.state_dict()[name])
        return diff, para, featurizer_i, classifier_i
    
    # ------------------ FedEM ------------------
    def local_train_fedem(self, global_models):
        device = next(global_models[0].parameters()).device
        for m in range(self.M):
            self.local_models[m].to(device)
            for name, param in global_models[m].state_dict().items():
                if not any(keyword in name for keyword in ["tracked", "running_mean", "running_var"]):
                    self.local_models[m].state_dict()[name].copy_(param.clone())
        criterion_none = nn.CrossEntropyLoss(reduction='none')
        optimizers = [
            torch.optim.Adam(
                self.local_models[m].parameters(), 
                lr=self.conf["lr"], 
                weight_decay=0.0
            ) for m in range(self.M)
        ]
        self.pi = self.pi.to(device)
        for m in range(self.M): 
            self.local_models[m].train()
        for e in range(self.conf["local_epochs"]):
            all_inputs, all_labels = [], []
            for inputs_b, labels_b in self.train_loader:
                all_inputs.append(inputs_b.to(device))
                all_labels.append(labels_b.to(device))
            all_inputs = torch.cat(all_inputs, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            with torch.no_grad():
                losses = torch.zeros(self.M, len(all_labels), device=device)
                for m in range(self.M):
                    self.local_models[m].eval()
                    outputs_all, _ = self.local_models[m](all_inputs)
                    losses[m] = criterion_none(outputs_all, all_labels)
                    self.local_models[m].train()
                log_pi = torch.log(self.pi + 1e-10).unsqueeze(1)
                samples_weights = torch.softmax(log_pi - losses, dim=0)
                self.pi = samples_weights.mean(dim=1)
            batch_loss_sum = 0
            for m in range(self.M):
                optimizers[m].zero_grad()
                outputs_all, _ = self.local_models[m](all_inputs)
                loss_vec = criterion_none(outputs_all, all_labels)
                loss = torch.mean(loss_vec * samples_weights[m])
                loss.backward()
                optimizers[m].step()
                batch_loss_sum += loss.item()
            if e % 10 == 0:
                print(f'Client {self.client_id} [FedEM], Epoch [{e}/{self.conf["local_epochs"]}], loss: {batch_loss_sum:.4f}')
        diffs = []
        for m in range(self.M):
            diff = dict()
            for name, data in self.local_models[m].state_dict().items():
                diff[name] = (data - global_models[m].state_dict()[name])
            diffs.append(diff)
        return diffs


    def save_item(self, model, global_protos=None, seed=None, dirspath=None):
        if dirspath is None:
            if seed is None:
                dirpath = f"./weights/{self.conf['datasets']}/{self.conf['aggregator']}"
            else:
                dirpath = f"./weights/{self.conf['datasets']}/{self.conf['aggregator']}_seed{seed}"
        else:
            dirpath = dirspath
        if not os.path.exists(dirpath):
            os.makedirs(dirpath, exist_ok=True)
        savepath_local = os.path.join(dirpath, f"local_{self.client_id}.pt")
        savepath_global = os.path.join(dirpath, f"global.pt")
        savepath_protos = os.path.join(dirpath, f"protos.pt")
        if self.conf["aggregator"] in ["ours", "ours_1", "ours_12", "ours_13"]:
            torch.save(self.local_model.state_dict(), savepath_local)
        elif self.conf["aggregator"] in ["fedproto", "ours_3"]:
            if global_protos is not None:
                torch.save(global_protos, savepath_protos)
            torch.save(self.local_model.state_dict(), savepath_local)
        elif self.conf["aggregator"] in ["fedavg", "fedprox", "scaffold", "fedrep", "fedwdp", "fedfocal",
                                         "fedavg_rwl", "fedprox_rwl", "scaffold_rwl", "fedrep_rwl", "fedwdp_rwl"]:
            if self.conf['aggregator'] in ["fedrep", "fedrep_rwl"]:
                for name, param in model.state_dict().items():
                    if not any(keyword in name for keyword in ["tracked", "running_mean", "running_var", "net.1"]):
                        self.local_model.state_dict()[name].copy_(param.clone())
            else:
                for name, param in model.state_dict().items():
                    if not any(keyword in name for keyword in ["tracked", "running_mean", "running_var"]):
                        self.local_model.state_dict()[name].copy_(param.clone())
            torch.save(self.local_model.state_dict(), savepath_local)
            torch.save(model.state_dict(), savepath_global)
        elif self.conf["aggregator"] in ["local", "pflda", "ditto",
                                         "local_rwl", "pflda_rwl", "ditto_rwl"]:
            torch.save(self.local_model.state_dict(), savepath_local)
        elif self.conf.get("aggregator") == "fedem":
            local_state = {
                'models': [m.state_dict() for m in self.local_models],
                'pi': self.pi.cpu()
            }
            torch.save(local_state, savepath_local)        


def wd_gaussian(mu1, cov1, mu2, cov2):
    FastMatSqrt = MPA_Lya.apply
    d = mu1.shape[-1]
    mean_term = torch.sum((mu1 - mu2) ** 2)
    cov1_expand = cov1.unsqueeze(0)
    cov2_expand = cov2.unsqueeze(0)
    cov_mm = torch.matmul(cov1_expand, cov2_expand)
    sqrt_term = FastMatSqrt(cov_mm)
    trace_term = torch.trace(cov1_expand[0]) + torch.trace(cov2_expand[0]) - 2 * torch.trace(sqrt_term[0])
    return mean_term + trace_term

def wd_gmm_per_class(mu, cov, MU_gmm, COV_gmm, PI_gmm):
    __device = mu.device
    K = PI_gmm.shape[0]
    sum_term = 0
    for k in range(K):
        mu_gmm_k = copy.deepcopy(MU_gmm[k].detach()).to(__device)
        cov_gmm_k = copy.deepcopy(COV_gmm[k].detach()).to(__device)
        pi_gmm_k = copy.deepcopy(PI_gmm[k].detach()).to(__device)
        wd_dist = wd_gaussian(mu, cov, mu_gmm_k, cov_gmm_k)
        sum_term += pi_gmm_k * wd_dist
    return sum_term


# https://github.com/yuetan031/fedproto/blob/main/lib/utils.py#L205
def agg_func(protos):
    for [label, proto_list] in protos.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            protos[label] = proto / len(proto_list)
        else:
            protos[label] = proto_list[0]
    return protos


class ContrastiveDistributionLoss(nn.Module):
    def __init__(self, sigma=1.0):
        super(ContrastiveDistributionLoss, self).__init__()
        self.sigma = sigma
    def forward(self, rep, proto_new, labels, Identity_G):
        unique_labels = torch.unique(labels)
        all_protos = []
        for label in unique_labels:
            idx = (labels == label).nonzero(as_tuple=True)[0][0]
            all_protos.append(copy.deepcopy(proto_new[idx]))
        all_protos = torch.stack(all_protos)
        rep_expanded = rep.unsqueeze(1)
        proto_expanded = all_protos.unsqueeze(0)
        diff = rep_expanded - proto_expanded
        logits = torch.zeros(rep.size(0), len(unique_labels), device=rep.device)
        for i, proto_label in enumerate(unique_labels):
            cov_iden = Identity_G[proto_label.item()]
            diff_i = diff[:, i, :]
            mahal_dist_sq = torch.sum(diff_i @ cov_iden * diff_i, dim=1)
            logits[:, i] = - mahal_dist_sq / (self.sigma * 2)
        targets = torch.zeros(rep.size(0), dtype=torch.long, device=rep.device)
        for i, label in enumerate(labels):
            targets[i] = torch.where(unique_labels == label)[0][0]
        loss = nn.functional.cross_entropy(logits, targets)
        return loss


class OrthonormalPrototypeLoss(nn.Module):
    def __init__(self):
        super(OrthonormalPrototypeLoss, self).__init__()
    def forward(self, rep, labels):
        device = rep.device
        unique_labels = torch.unique(labels)
        class_protos = []
        for label in unique_labels:
            indices = (labels == label)
            if not indices.any():
                continue
            class_rep = rep[indices].mean(dim=0)
            norm = torch.norm(class_rep, p=2)
            if norm > 0:
                class_rep = class_rep / norm
            class_protos.append(class_rep)
        P = torch.stack(class_protos, dim=1)
        PTP = torch.matmul(P.t(), P)
        I = torch.eye(PTP.size(0), device=device)
        diff = PTP - I
        ortho_loss = torch.norm(diff, p='fro')**2
        return ortho_loss


# For `FedWDP`
def sliced_wasserstein_distance(w_local, w_global, num_projections=50, p=2, device=None):
    if device is None:
        device = w_local.device
    if w_local.dim() == 1:
        w_local = w_local.unsqueeze(1)
    if w_global.dim() == 1:
        w_global = w_global.unsqueeze(1)
    N, d = w_local.shape
    projections = torch.randn(d, num_projections, device=device)
    projections = projections / torch.norm(projections, dim=0, keepdim=True)
    proj_local = torch.matmul(w_local, projections)
    proj_global = torch.matmul(w_global, projections)
    proj_local_sorted, _ = torch.sort(proj_local, dim=0)
    proj_global_sorted, _ = torch.sort(proj_global, dim=0)
    if p == 1:
        distance = torch.abs(proj_local_sorted - proj_global_sorted)
    elif p == 2:
        distance = (proj_local_sorted - proj_global_sorted) ** 2
    else:
        distance = torch.abs(proj_local_sorted - proj_global_sorted) ** p
    swd = torch.mean(distance)
    if p != 1 and p != 2:
        swd = torch.pow(swd, 1.0 / p)
    return swd

# For `FedFocal`
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss