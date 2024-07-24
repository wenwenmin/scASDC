import numpy as np
from munkres import Munkres, print_matrix
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from scipy.optimize import linear_sum_assignment as linear
from sklearn import metrics
import copy
from tqdm import tqdm
import torch
import torch.nn as nn
from utils_ProtoMGAE import create_optimizer, accuracy
from sklearn.cluster import KMeans

torch.set_num_threads(1)


def cluster_acc(y_true, y_pred):
    y_true = y_true - np.min(y_true)

    l1 = list(set(y_true))
    numclass1 = len(l1)

    l2 = list(set(y_pred))
    numclass2 = len(l2)

    ind = 0
    if numclass1 != numclass2:
        for i in l1:
            if i in l2:
                pass
            else:
                y_pred[ind] = i
                ind += 1

    l2 = list(set(y_pred))
    numclass2 = len(l2)

    cost = np.zeros((numclass1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)

    # match two clustering results by Munkres algorithm
    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)

    # get the match results
    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        # correponding label in l2:
        c2 = l2[indexes[i][1]]

        # ai is the index with label==c2 in the pred_label list
        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c

    acc = metrics.accuracy_score(y_true, new_predict)
    f1_macro = metrics.f1_score(y_true, new_predict, average='macro')
    return acc, f1_macro


def eva(y_true, y_pred, epoch=0):
    acc, f1 = cluster_acc(y_true, y_pred)
    nmi = nmi_score(y_true, y_pred, average_method='arithmetic')
    ari = ari_score(y_true, y_pred)
    print(epoch, ':acc {:.4f}'.format(acc), ', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari),
          ', f1 {:.4f}'.format(f1))
    return acc, nmi, ari, f1


def node_clustering(model, graph, x, num_classes, epoch, device, y):
    model.eval()

    with torch.no_grad():
        emb = model.embed(graph.to(device), x.to(device))

    kmeans = KMeans(n_clusters=num_classes, n_init=20)
    print(kmeans)
    labels = y
    y_pred = kmeans.fit_predict(emb.cpu().numpy())
    clu_acc = eva(labels, y_pred, epoch)
    return clu_acc


def node_clustering_old(model, graph, x, num_classes, epoch, device):
    model.eval()

    with torch.no_grad():
        emb = model.embed(graph.to(device), x.to(device))

    kmeans = KMeans(n_clusters=num_classes, n_init=20)
    print(kmeans)
    labels = graph.ndata["label"].cpu().numpy()
    y_pred = kmeans.fit_predict(emb.cpu().numpy())
    clu_acc = eva(labels, y_pred, epoch)
    return clu_acc


def node_classification_evaluation(model, dataset, graph, x, num_classes, lr_f, weight_decay_f, max_epoch_f, device,
                                   linear_prob=True, mute=False):
    model.eval()
    if linear_prob:
        with torch.no_grad():
            x = model.embed(graph.to(device), x.to(device))
            in_feat = x.shape[1]
        encoder = LogisticRegression(in_feat, num_classes)
    else:
        encoder = model.encoder
        encoder.reset_classifier(num_classes)

    num_finetune_params = [p.numel() for p in encoder.parameters() if p.requires_grad]
    if not mute:
        print(f"num parameters for finetuning: {sum(num_finetune_params)}")

    encoder.to(device)
    optimizer_f = create_optimizer("adam", encoder, lr_f, weight_decay_f)
    final_acc, estp_acc = linear_probing_for_transductive_node_classiifcation(encoder, dataset, graph, x, optimizer_f,
                                                                              max_epoch_f, device, mute)
    return final_acc, estp_acc


def linear_probing_for_transductive_node_classiifcation(model, dataset, graph, feat, optimizer, max_epoch, device,
                                                        mute=False):
    criterion = torch.nn.CrossEntropyLoss()

    graph = graph.to(device)
    x = feat.to(device)

    cla_acc = []
    es_cla_acc = []
    for i in range(20):
        if dataset == 'wikics':
            train_mask = graph.ndata["train_mask"][:, i].bool()
            val_mask = graph.ndata["val_mask"][:, i].bool()
            test_mask = graph.ndata["test_mask"].bool()
        else:
            train_mask = graph.ndata["train_mask"]
            val_mask = graph.ndata["val_mask"]
            test_mask = graph.ndata["test_mask"]
        labels = graph.ndata["label"]

        best_val_acc = 0
        best_val_epoch = 0
        best_model = None

        for epoch in range(max_epoch):
            model.train()
            out = model(graph, x)
            loss = criterion(out[train_mask], labels[train_mask])
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3)
            optimizer.step()

            with torch.no_grad():
                model.eval()
                pred = model(graph, x)
                val_acc = accuracy(pred[val_mask], labels[val_mask])
                val_loss = criterion(pred[val_mask], labels[val_mask])
                test_acc = accuracy(pred[test_mask], labels[test_mask])
                test_loss = criterion(pred[test_mask], labels[test_mask])

            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                best_val_epoch = epoch
                best_model = copy.deepcopy(model)

        best_model.eval()
        with torch.no_grad():
            pred = best_model(graph, x)
            estp_test_acc = accuracy(pred[test_mask], labels[test_mask])
        cla_acc.append(test_acc)
        es_cla_acc.append(estp_test_acc)

    # (final_acc, es_acc, best_acc)
    # return test_acc, estp_test_acc
    return np.mean(cla_acc), np.mean(es_cla_acc)


def linear_probing_for_inductive_node_classiifcation(model, x, labels, mask, optimizer, max_epoch, device, mute=False):
    if len(labels.shape) > 1:
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()
    train_mask, val_mask, test_mask = mask

    best_val_acc = 0
    best_val_epoch = 0
    best_model = None

    if not mute:
        epoch_iter = tqdm(range(max_epoch))
    else:
        epoch_iter = range(max_epoch)

    for epoch in range(max_epoch):
        model.train()
        out = model(None, x)
        loss = criterion(out[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3)
        optimizer.step()

        with torch.no_grad():
            model.eval()
            pred = model(None, x)
            val_acc = accuracy(pred[val_mask], labels[val_mask])
            val_loss = criterion(pred[val_mask], labels[val_mask])
            test_acc = accuracy(pred[test_mask], labels[test_mask])
            test_loss = criterion(pred[test_mask], labels[test_mask])

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_val_epoch = epoch
            best_model = copy.deepcopy(model)

        if not mute:
            epoch_iter.set_description(
                f"# Epoch: {epoch}, train_loss:{loss.item(): .4f}, val_loss:{val_loss.item(): .4f}, val_acc:{val_acc}, test_loss:{test_loss.item(): .4f}, test_acc:{test_acc: .4f}")

    best_model.eval()
    with torch.no_grad():
        pred = best_model(None, x)
        estp_test_acc = accuracy(pred[test_mask], labels[test_mask])
    if mute:
        print(
            f"# IGNORE: --- TestAcc: {test_acc:.4f}, early-stopping-TestAcc: {estp_test_acc:.4f}, Best ValAcc: {best_val_acc:.4f} in epoch {best_val_epoch} ")
    else:
        print(
            f"--- TestAcc: {test_acc:.4f}, early-stopping-TestAcc: {estp_test_acc:.4f}, Best ValAcc: {best_val_acc:.4f} in epoch {best_val_epoch}")

    return test_acc, estp_test_acc


class LogisticRegression(nn.Module):
    def __init__(self, num_dim, num_class):
        super().__init__()
        self.linear = nn.Linear(num_dim, num_class)

    def forward(self, g, x, *args):
        logits = self.linear(x)
        return logits
