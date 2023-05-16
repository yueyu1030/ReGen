import faiss 
import numpy as np 
import os 
import matplotlib 
matplotlib.use("Agg")
import matplotlib.pyplot as plt 
import json 

def inference_cal(train_pred, train_feat, train_label,  unlabeled_pred, unlabeled_feat, unlabeled_label, unlabeled_pseudo,k,  gamma = 0.1, beta=0.1, prev_val = None):
    train_pred = np.array(train_pred)
    unlabeled_pred = np.array(unlabeled_pred)
    d = train_feat.shape[-1]
    index = faiss.IndexFlatL2(d)
    index.add(train_feat)
    D, I = index.search(unlabeled_feat, k)
    unlabeled_pred =  np.expand_dims(unlabeled_pred, axis = 1)
    # [#unlabel, 1]
    # train_pred[I] ---> [#unlabel, k]
    # print(unlabeled_pred.shape)
    score = np.log((1e-10 + train_pred[I])/ (1e-10 + unlabeled_pred)) * train_pred[I]
    # print(score.shape)
    mean_kl = np.mean(np.sum(score, axis = -1), axis = -1)

    # mean_mse =  np.mean((train_pred[I] - unlabeled_pred)**2, axis = -1)
    # train pred (n_samples, n_class)
    # train pred[I] (n_samples, n_neighbor, n_class)
    mean_pred = np.mean(train_pred[I], axis = 1)
    epsilon = 1e-13
    # Calculating entropy across multiple MCD forward passes 
    entropy = -np.sum(mean_pred * np.log(mean_pred + epsilon), axis=-1) # shape (n_samples,)
    mutual_info = entropy - np.mean(np.sum(-train_pred[I] * np.log(train_pred[I] + epsilon),
                                                axis=-1), axis=1) # shape (n_samples,)
    var_mse =  np.var(train_pred[I], axis = -1)
    # np.log((1e-10+train_pred[I])/ (1e-10+unlabeled_pred)) * train_pred[I]
    # print(score.shape)
    #  = np.mean(np.sum(score, axis = -1), axis = -1)
    # print(mean_mse, var_mse)
    # 1 for cls, 0.1 for reg
    if prev_val is not None:
        current_val = prev_val * gamma + (1- gamma) * (mean_kl + mutual_info * beta)
    else:
        current_val = mean_kl + mutual_info * beta
    idx = np.argsort(current_val)

    unlabel_correct = [1 if x == y else 0 for (x, y) in zip(unlabeled_pseudo, unlabeled_label)]
    sorted_acc = np.array(unlabel_correct)[idx]
    sorted_mean = np.cumsum(sorted_acc)/(1+np.arange(sorted_acc.shape[0]))
    return idx, sorted_acc, sorted_mean, current_val

def inference_conf(train_pred, train_feat, train_label,  unlabeled_pred, unlabeled_feat, unlabeled_label, unlabeled_pseudo, gamma = 0.1, prev_val = None):
    train_pred = np.array(train_pred)
    unlabeled_pred = np.array(unlabeled_pred)
    current_val = -np.max(unlabeled_pred, axis = -1)
    if prev_val is not None:
        current_val = prev_val * gamma + (1- gamma) * (current_val)
    else:
        current_val = current_val
    idx = np.argsort(current_val)

    unlabel_correct = [1 if x == y else 0 for (x, y) in zip(unlabeled_pseudo, unlabeled_label)]
    sorted_acc = np.array(unlabel_correct)[idx]
    sorted_mean = np.cumsum(sorted_acc)/(1+np.arange(sorted_acc.shape[0]))
    return idx, sorted_acc, sorted_mean, current_val

def inference_uncertainty(unlabeled_label, unlabeled_pseudo, mutual_info, gamma = 0.1, prev_val = None):
    if prev_val is not None:
        current_val = prev_val * gamma + (1- gamma) * (mutual_info)
    else:
        current_val = mutual_info
    idx = np.argsort(current_val)
    unlabel_correct = [1 if x == y else 0 for (x, y) in zip(unlabeled_pseudo, unlabeled_label)]
    sorted_acc = np.array(unlabel_correct)[idx]
    sorted_mean = np.cumsum(sorted_acc)/(1+np.arange(sorted_acc.shape[0]))
    return idx, sorted_acc, sorted_mean, current_val

def save_data(train_pred, train_feat, train_label,  unlabeled_pred, unlabeled_feat, unlabeled_label, unlabeled_pseudo, mutual_info_bald = None, dataset = 'agnews', n_iter = 0):
    # if n_iter == 0:
    #     path = f"{dataset}/"
        
    # else:
    path = f"{dataset}/{n_iter}"
    os.makedirs(path, exist_ok = True)
    
    with open(f"{path}/train_pred.npy", 'wb') as f:
        np.save(f, train_pred)
    
    with open(f"{path}/train_feat.npy", 'wb') as f:
        np.save(f, train_feat)
    
    with open(f"{path}/train_label.npy", 'wb') as f:
        np.save(f, train_label)

    with open(f"{path}/unlabeled_pred.npy", 'wb') as f:
        np.save(f, unlabeled_pred)

    with open(f"{path}/unlabeled_feat.npy", 'wb') as f:
        np.save(f, unlabeled_feat)
    
    with open(f"{path}/unlabeled_label.npy", 'wb') as f:
        np.save(f, unlabeled_label)
    
    with open(f"{path}/unlabeled_pseudo.npy", 'wb') as f:
        np.save(f, unlabeled_pseudo)
    if mutual_info_bald is not None:
        with open(f"{path}/mutual_info_bald.npy", 'wb') as f:
            np.save(f, mutual_info_bald)

def plot_data(dataset = 'agnews', n_labels = 30, n_iter = 0, k = 5, ckpt = 'news_ckpt', prompt_id = 1, topN=50):
    if n_iter == 0:
        path = f"{dataset}/{n_labels}"
    else:
        path = f"{dataset}/{n_labels}_{n_iter}"

    train_pred = np.load(f"{path}/train_pred.npy")
    train_feat = np.load(f"{path}/train_feat.npy")
    train_label = np.load(f"{path}/train_label.npy")

    unlabeled_pred = np.load(f"{path}/unlabeled_pred.npy")
    unlabeled_feat = np.load(f"{path}/unlabeled_feat.npy")
    unlabeled_label = np.load(f"{path}/unlabeled_label.npy")
    unlabeled_pseudo = np.load(f"{path}/unlabeled_pseudo.npy")

    # train_pred = np.array(train_pred)
    # unlabeled_pred = np.array(unlabeled_pred)
    # d = train_feat.shape[-1]
    # index = faiss.IndexFlatL2(d)
    # index.add(train_feat)
    # D, I = index.search(unlabeled_feat, k)
    # unlabeled_pred =  np.expand_dims(unlabeled_pred, axis = 1)
    # # [#unlabel, 1]
    # # train_pred[I] ---> [#unlabel, k]
    # # print(unlabeled_pred.shape)
    # print(train_pred.shape, I.shape, ((1e-10 + train_pred[I])/ (1e-10 + unlabeled_pred)).shape, unlabeled_pred.shape, unlabeled_feat.shape, train_feat.shape)
    # score = np.log((1e-10 + train_pred[I])/ (1e-10 + unlabeled_pred)) * train_pred[I]
    # mean_kl = np.mean(np.sum(score, axis = -1), axis = -1)
    idx_cal, sorted_acc_cal, sorted_mean_cal, current_val = inference_cal(train_pred, train_feat, train_label,  unlabeled_pred, unlabeled_feat, unlabeled_label, unlabeled_pseudo, k = k, gamma = 0.0, beta=0.1, prev_val = None)
    idx_conf, sorted_acc_conf, sorted_mean_conf, current_val = inference_conf(train_pred, train_feat, train_label,  unlabeled_pred, unlabeled_feat, unlabeled_label, unlabeled_pseudo, gamma = 0.1, prev_val = None)
    
    plt.figure(figsize = [12, 7])
    plt.plot(sorted_mean_cal[:30000])
    plt.plot(sorted_mean_conf[:30000])
    plt.ylim([0.8, 1])
    plt.legend([f'cal k={k}', 'conf'])
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{path}/{n_labels}_{n_iter}_{k}.pdf")
    train_pred_pseudo = np.argmax(train_pred, axis = -1)
    from collections import Counter 
    x = Counter(train_pred_pseudo)
    print(np.mean(train_pred_pseudo == train_label), x)
    class_0_id =  np.argsort(train_pred[:, 0], axis = -1)
    class_1_id =  np.argsort(train_pred[:, 1], axis = -1)
    class_2_id =  np.argsort(train_pred[:, 2], axis = -1)
    class_3_id =  np.argsort(train_pred[:, 3], axis = -1)
    print(class_0_id[1], train_pred[class_0_id[1]])
    print(class_0_id[2], train_pred[class_0_id[2]])
    print(class_0_id[3], train_pred[class_0_id[3]])
    idx_lst = []
    for i in range(train_pred.shape[-1]):
        idx_lst_tmp = []
        class_id = np.argsort(train_pred[:, i], axis = -1)[::-1]
        for j in class_id[:120]:
            if train_pred_pseudo[i] == train_label[i]:
                idx_lst_tmp.append(j)
        idx_lst = idx_lst + idx_lst_tmp
    print(idx_lst)
    text = []
    with open(f"/localscratch/yueyu/datasets/{dataset}_openws/filter_data/{ckpt}_all_top{topN}_{prompt_id}_round{n_iter}.jsonl", "r") as f:
        for lines in f:
            text.append(json.loads(lines))
    os.makedirs(f"/localscratch/yueyu/datasets/{dataset}_openws/retrieve_data", exist_ok=True)
    
    with open(f"/localscratch/yueyu/datasets/{dataset}_openws/retrieve_data/{ckpt}_all_top{topN}_{prompt_id}_round{n_iter}.jsonl", "w") as f:
        for i in idx_lst:
            f.write(json.dumps(text[i]) + '\n')

    # print()




def load_pred_data(dataset = 'agnews', n_labels = 10, n_iter = 0):
    # os.makedirs(f"{dataset}/{n_labels}", exist_ok = True)
    # with open(f"{dataset}/{n_labels}/train_pred.npy", 'rb') as f:
    if n_iter == 0:
        path = f"{dataset}/{n_labels}"
    else:
        path = f"{dataset}/{n_labels}_{n_iter}"
    train_pred = np.load(f"{path}/train_pred.npy")

    train_feat = np.load(f"{path}/train_feat.npy")

    train_label = np.load(f"{path}/train_label.npy")

    unlabeled_pred = np.load(f"{path}/unlabeled_pred.npy")

    unlabeled_feat = np.load(f"{path}/unlabeled_feat.npy")

    unlabeled_label = np.load(f"{path}/unlabeled_label.npy")
    
    unlabeled_pseudo = np.load(f"{path}/unlabeled_pseudo.npy")
    try:
        mutual_info_bald = np.load(f"{path}/mutual_info_bald.npy")
    except:
        mutual_info_bald = None 
    return train_pred, train_feat, train_label,  unlabeled_pred, unlabeled_feat, unlabeled_label, unlabeled_pseudo, mutual_info_bald


def create_new_dataset(task, init_label, final_label, idx_cal, args, train_label, unlabeled_label, unlabeled_pred):
    import random
    import json     
    select = idx_cal[:args.num_unlabeled]
    # print(len(select), idx_cal.shape)
    # exit()
    labeled_idx = []
    al_idx = []
    from collections import Counter
    x = Counter(train_label)
    print(x, idx_cal.shape)
    unlabeled_idx = []
    n_label = final_label
    for i in idx_cal[::-1]:
        class_i = unlabeled_label[i]
        if x[class_i] < (n_label) and random.random() > 0.2:
            unlabeled_idx.append(int(i))
            # print(unlabeled_pred[i])
            x[class_i] += 1
    print(len(unlabeled_idx), x)
    with open(f"agnews_{n_label}.json", 'w') as f:
        json.dump(unlabeled_idx, f)
    os.makedirs(f"../datasets/{task}-{final_label}-10/", exist_ok = True)
    
    labeled_data = []
    with open(f"../datasets/{task}-{init_label}-10/train.json", 'r') as f:
        for lines in f:
            labeled_data.append(json.loads(lines.strip()))
    print(len(labeled_data))

    unlabeled_data = []
    with open(f"../datasets/{task}-{init_label}-10/unlabeled.json", 'r') as f:
        for i, lines in enumerate(f):
            if i in unlabeled_idx:
                labeled_data.append(json.loads(lines.strip()))
            else:
                unlabeled_data.append(json.loads(lines.strip()))

    print(len(labeled_data), len(unlabeled_data))

    with open(f"../datasets/{task}-{final_label}-10/train.json", 'w') as f:
        for x in labeled_data:
            f.write(json.dumps(x) + "\n")

    with open(f"../datasets/{task}-{final_label}-10/unlabeled.json", 'w') as f:
        for x in unlabeled_data:
            f.write(json.dumps(x) + "\n")
