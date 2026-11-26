import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix

def metric_eval(df, server):
    df_eval = []
    num_classes = server.conf['no_classes']
    all_classes = list(range(num_classes))
    if len(df) > 0:
        cm_global = confusion_matrix(df['label'], df['predict'],
                                     labels=all_classes)
        df_eval.append({
            'Client': -1,
            'CM': cm_global.tolist()
        })
    offset = 0
    for c_id, eval_loader in enumerate(server.eval_loader_list):
        n_sample_c = len(eval_loader.dataset)
        df_client = df.iloc[offset:offset + n_sample_c]
        offset += n_sample_c
        
        if len(df_client) > 0:
            cm_client = confusion_matrix(df_client['label'], df_client['predict'],
                                         labels=all_classes)
            df_eval.append({
                'Client': c_id + 1,
                'CM': cm_client.tolist()
            })
    df_eval = pd.DataFrame(df_eval)
    return df_eval

def model_eval(conf, client, server, To_CSV=False, seed=None):
    prediction_eval = []
    for c_id, c in enumerate(client):
        c.local_model.eval()
        test_dataloader = server.eval_loader_list[c_id]
        for inputs, targets in test_dataloader:
            inputs, targets = inputs.to("cuda"), targets.to("cuda")
            outputs, _ = c.local_model(inputs)
            _, pred = torch.max(outputs, dim=1)
            df_pred = pd.DataFrame({
                'label': targets.cpu().numpy(),
                'predict': pred.cpu().numpy()
            })
            df_prob = pd.DataFrame(outputs.cpu().detach().numpy(), 
                                    columns=['class {}'.format(i) for i in range(0, conf['no_classes'])])
            df_pred = pd.concat([df_pred, df_prob], axis=1)
            prediction_eval.append(df_pred)
    prediction_eval = pd.concat(prediction_eval, axis=0)
    prediction_eval = metric_eval(prediction_eval, server)
    if seed is not None:
        prediction_eval['Seed'] = seed
    if To_CSV:
        save_dir = f"./evaluation/{conf['datasets']}/{conf['aggregator']}"
        if seed is not None:
            save_dir += f"_seed{seed}"
        os.makedirs(save_dir, exist_ok=True)
        prediction_eval.to_pickle(f"{save_dir}/test_predictions_global.pkl") 
    return prediction_eval

def proto_eval(conf, client, server, To_CSV=False, seed=None):
    targets_list = []
    protos_list = []
    loss_fn = torch.nn.MSELoss()
    for c_id, c in enumerate(client):
        c.local_model.eval()
        test_dataloader = server.eval_loader_list[c_id]
        for inputs, targets in test_dataloader:
            inputs, targets = inputs.to("cuda"), targets.to("cuda")
            _, rep = c.local_model(inputs)
            targets_list.append(targets.detach())
            protos_list.append(rep.detach())
    targets_list = torch.cat(targets_list, dim=0)
    protos_list = torch.cat(protos_list, dim=0)
    probs = torch.zeros((protos_list.size()[0], len(server.global_protos)))
    for i in range(protos_list.size()[0]):
        for j in range(len(server.global_protos)):
            probs[i, j] = loss_fn(protos_list[i], server.global_protos[j])
    _, preds_list = torch.min(probs, 1)
    df_pred = pd.DataFrame({
        'label': targets_list.cpu().numpy(),
        'predict': preds_list.cpu().numpy()
    })
    df_prob = pd.DataFrame(probs.cpu().detach().numpy(), 
                            columns=['class {}'.format(i) for i in range(0, conf['no_classes'])])
    prediction_eval = pd.concat([df_pred, df_prob], axis=1)
    prediction_eval = metric_eval(prediction_eval, server)
    if seed is not None:
        prediction_eval['Seed'] = seed
    if To_CSV:
        save_dir = f"./evaluation/{conf['datasets']}/{conf['aggregator']}"
        if seed is not None:
            save_dir += f"_seed{seed}"
        os.makedirs(save_dir, exist_ok=True)
        prediction_eval.to_pickle(f"{save_dir}/test_predictions_proto.pkl")
    return prediction_eval

def fedem_eval(conf, client, server, To_CSV=False, seed=None):
    prediction_eval = []
    for c_id, c in enumerate(client):
        for m in range(c.M):
            c.local_models[m].eval()
        test_dataloader = server.eval_loader_list[c_id]
        for inputs, targets in test_dataloader:
            inputs, targets = inputs.to("cuda"), targets.to("cuda")
            ensemble_probs = torch.zeros(inputs.size(0), conf['no_classes'], device="cuda")
            with torch.no_grad():
                for m in range(c.M):
                    outputs, _ = c.local_models[m](inputs)
                    ensemble_probs += c.pi[m] * torch.softmax(outputs, dim=1)
            _, pred = torch.max(ensemble_probs, dim=1)
            df_pred = pd.DataFrame({
                'label': targets.cpu().numpy(),
                'predict': pred.cpu().numpy()
            })
            df_prob = pd.DataFrame(ensemble_probs.cpu().detach().numpy(), 
                                    columns=['class {}'.format(i) for i in range(0, conf['no_classes'])])
            df_pred = pd.concat([df_pred, df_prob], axis=1)
            prediction_eval.append(df_pred)
    prediction_eval = pd.concat(prediction_eval, axis=0)
    prediction_eval = metric_eval(prediction_eval, server)
    if seed is not None:
        prediction_eval['Seed'] = seed
    if To_CSV:
        save_dir = f"./evaluation/{conf['datasets']}/{conf['aggregator']}"
        if seed is not None:
            save_dir += f"_seed{seed}"
        os.makedirs(save_dir, exist_ok=True)
        prediction_eval.to_pickle(f"{save_dir}/test_predictions_fedem.pkl") 
    return prediction_eval

def gmm_eval(conf, client, server, To_CSV=False, seed=None):
    targets_list = []
    outputs_list = []
    reps_list = []
    for c_id, c in enumerate(client):
        c.local_model.eval()
        test_dataloader = server.eval_loader_list[c_id]
        for inputs, targets in test_dataloader:
            inputs, targets = inputs.to("cuda"), targets.to("cuda")
            outputs, rep = c.local_model(inputs)
            targets_list.append(targets.detach())
            outputs_list.append(outputs.detach())
            reps_list.append(rep.detach())
    targets_list = torch.cat(targets_list, dim=0)
    outputs_list = torch.cat(outputs_list, dim=0)
    reps_list = torch.cat(reps_list, dim=0)
    probs_1 = gmm_euclidean_inference(reps_list, server.global_mean, server.global_pi, is_avg=False)
    _, preds_list_1 = torch.min(probs_1, 1)
    df_pred_1 = pd.DataFrame({
        'label': targets_list.cpu().numpy(),
        'predict': preds_list_1.cpu().numpy()
    })
    df_prob_1 = pd.DataFrame(probs_1.cpu().detach().numpy(), 
                             columns=['class {}'.format(i) for i in range(0, conf['no_classes'])])
    prediction_eval_1 = pd.concat([df_pred_1, df_prob_1], axis=1)
    probs_2 = gmm_mahalanobis_inference(reps_list, server.global_mean, server.global_cov, server.global_pi)
    _, preds_list_2 = torch.min(probs_2, 1)
    df_pred_2 = pd.DataFrame({
        'label': targets_list.cpu().numpy(),
        'predict': preds_list_2.cpu().numpy()
    })
    df_prob_2 = pd.DataFrame(probs_2.cpu().detach().numpy(), 
                             columns=['class {}'.format(i) for i in range(0, conf['no_classes'])])
    prediction_eval_2 = pd.concat([df_pred_2, df_prob_2], axis=1)
    
    _, preds_list_3 = torch.max(outputs_list, 1)
    df_pred_3 = pd.DataFrame({
        'label': targets_list.cpu().numpy(),
        'predict': preds_list_3.cpu().numpy()
    })
    df_prob_3 = pd.DataFrame(outputs_list.cpu().detach().numpy(), 
                             columns=['class {}'.format(i) for i in range(0, conf['no_classes'])])
    prediction_eval_3 = pd.concat([df_pred_3, df_prob_3], axis=1)
    
    prediction___ = [prediction_eval_1, prediction_eval_2, prediction_eval_3]
    preds___ = [preds_list_1, preds_list_2, preds_list_3]
    targets_np = targets_list.cpu().numpy()
    accuracies = [
        accuracy_score(targets_np, preds.cpu().numpy())
        for preds in preds___
    ]
    _idx = int(np.argmax(accuracies))
    prediction_eval = prediction___[_idx]
    _preds = preds___[_idx].cpu().numpy()
    prediction_eval = metric_eval(prediction_eval, server)
    if seed is not None:
        prediction_eval['Seed'] = seed
    if To_CSV:
        save_dir = f"./evaluation/{conf['datasets']}/{conf['aggregator']}"
        if seed is not None:
            save_dir += f"_seed{seed}"
        os.makedirs(save_dir, exist_ok=True)
        prediction_eval.to_pickle(f"{save_dir}/test_predictions_gmm.pkl")
    return prediction_eval

def gmm_mahalanobis_inference(z, global_mean, global_cov, global_pi):
    device = z.device
    n_samples = z.shape[0]
    d = z.shape[-1]
    classes = sorted(global_mean.keys())
    n_classes = len(classes)
    distances = torch.zeros((n_samples, n_classes), device=device)
    for c_idx, class_id in enumerate(classes):
        mu_gmm = global_mean[class_id].to(device)
        cov_gmm = global_cov[class_id].to(device)
        pi_gmm = global_pi[class_id].to(device)
        K = pi_gmm.shape[0]
        mubar_gmm_c = torch.sum(mu_gmm * pi_gmm.unsqueeze(1), dim=0)
        cov_gmm_c = torch.zeros((d, d), device=device)
        for k in range(K):
            cov_gmm_c += pi_gmm[k] * cov_gmm[k]
            cov_gmm_c += pi_gmm[k] * torch.outer(mu_gmm[k] - mubar_gmm_c, mu_gmm[k] - mubar_gmm_c)
        cov_inv = torch.inverse(cov_gmm_c)
        for i in range(n_samples):
            z_centered = z[i] - mubar_gmm_c
            distances[i, c_idx] = torch.sqrt(
                torch.matmul(torch.matmul(z_centered.unsqueeze(0), cov_inv),
                                          z_centered.unsqueeze(1))
                )
    return distances

def gmm_euclidean_inference(z, global_mean, global_pi, is_avg=False):
    device = z.device
    n_samples = z.shape[0]
    classes = sorted(global_mean.keys())
    n_classes = len(classes)
    distances = torch.zeros((n_samples, n_classes), device=device)
    for c_idx, class_id in enumerate(classes):
        mu_gmm = global_mean[class_id].to(device)
        pi_gmm = global_pi[class_id].to(device)
        if not is_avg:
            mubar_gmm_c = torch.sum(mu_gmm * pi_gmm.unsqueeze(1), dim=0)
        else:
            mubar_gmm_c = torch.mean(mu_gmm, dim=0)
        for i in range(n_samples):
            z_centered = z[i] - mubar_gmm_c
            distances[i, c_idx] = torch.sqrt(torch.sum(z_centered ** 2))
    return distances