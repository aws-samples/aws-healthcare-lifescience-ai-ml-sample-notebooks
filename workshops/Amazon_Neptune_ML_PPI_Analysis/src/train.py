# Adapted from
# https://github.com/awslabs/neptuneml-toolkit/blob/main/examples/custom-models/introduction/movie-lens-rgcn/link-predict/src/old_train.py
import argparse
import os
import time
from functools import partial
from neptuneml_toolkit.train import (
    get_training_config,
    get_train_eids,
    get_valid_eids,
    get_all_forward_edges,
)
from neptuneml_toolkit.metrics import save_eval_metrics, mrr
from neptuneml_toolkit.transform import (
    save_link_prediction_model_artifacts,
    normalize_hyperparameter_keys,
)
from neptuneml_toolkit.utils import get_device_type
from neptuneml_toolkit.graphloader import GraphLoader
from neptuneml_toolkit.modelzoo import (
    RGCNEncoder,
    MLPFeatureTransformer,
    GraphDistmultDecoder,
    DistmultDecoder,
)
from sklearn.metrics import roc_auc_score
import sys
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class RGCNLinkPrediction(nn.Module):
    def __init__(self, etypes, in_sizes, hidden_size, num_bases, num_encoder_layers):
        super(RGCNLinkPrediction, self).__init__()
        self.feature_transformer = MLPFeatureTransformer(
            in_sizes, hidden_size, per_feat_name=False
        )
        self.encoder = RGCNEncoder(
            etypes, hidden_size, hidden_size, num_encoder_layers, num_bases=num_bases
        )
        self.decoder = GraphDistmultDecoder(hidden_size, len(etypes))

    def forward(self, g, x, pos_graph, neg_graph):
        h = self.feature_transformer(x)
        embeddings = self.encoder(g, h)
        return self.decoder(pos_graph, embeddings), self.decoder(neg_graph, embeddings)

    @torch.no_grad()
    def encode(self, g, x):
        h = self.feature_transformer(x)
        return self.encoder(g, h)

    @torch.no_grad()
    def score(self, u, v, etype_ids):
        r_emb = self.decoder.w_relation.clone()[etype_ids.long()]
        return torch.sum(u * r_emb * v, dim=-1)

    def get_embeddings(self, g, x, batch_size, device="cpu", num_workers=0):
        h = self.feature_transformer(x)
        return self.encoder.batch_inference(
            g, h, batch_size, device=device, num_workers=num_workers
        )

    def save(self, model_file):
        torch.save({"model_state_dict": self.state_dict()}, model_file)


def get_model(g, in_size, hyperparameters, model_file=None, device="cpu"):
    model = RGCNLinkPrediction(
        g.etypes,
        in_size,
        int(hyperparameters["hidden_size"]),
        int(hyperparameters["num_bases"]),
        int(hyperparameters["num_encoder_layers"]),
    )
    if model_file is not None:
        model_dict = torch.load(model_file, map_location=torch.device("cpu"))
        model.load_state_dict(model_dict["model_state_dict"])

    model = model.to(device)
    return model


def evaluate(model, features, loss_fn, dataloader, masked_edges, device="cpu"):

    model.eval()
    metrics = []
    losses = []
    for i, (input_nodes, pos_graph, neg_graph, subgraphs) in enumerate(dataloader):
        batch_features = {
            ntype: {
                feat_name: feat[node_idx].to(device)
                for feat_name, feat in features[ntype].items()
            }
            for ntype, node_idx in input_nodes.items()
        }
        subgraphs = [subgraph.to(device) for subgraph in subgraphs]
        h = model.encode(subgraphs, batch_features)
        pos_scores, mrr_metric = get_scores(
            pos_graph, h, model.score, calc_mrr=True, edge_mask=masked_edges
        )
        neg_scores, _ = get_scores(neg_graph, h, model.score)
        predictions = torch.cat([pos_scores, neg_scores])
        labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])
        losses.append(loss_fn(predictions, labels).item())
        metrics.append(mrr_metric)

    metric = np.average(metrics)
    loss = np.average(losses)

    return metric, loss


def get_scores(g, h, scoring_fn, calc_mrr=False, edge_mask=None):
    scores, mrrs = [], []
    for etype_id, etype in enumerate(g.canonical_etypes):
        if g.num_edges(etype):
            src_type, rel_type, dst_type = etype
            u, v = g.edges(etype=etype)
            etype_ids = torch.full_like(u, etype_id)
            scores.append(scoring_fn(h[src_type][u], h[dst_type][v], etype_ids))
            if calc_mrr:
                mrr_score, _ = mrr(
                    h,
                    (u, v, etype_ids),
                    scoring_fn,
                    masked_edges=edge_mask,
                    src_type=src_type,
                    dst_type=dst_type,
                )
                mrrs.append(mrr_score)
    mrr_metric = np.average(mrrs) if calc_mrr else None
    return torch.cat(scores), mrr_metric


def train_n_epochs(
    model,
    optimizer,
    features,
    loss_fn,
    train_dataloader,
    validation_dataloader,
    eval_edge_mask,
    n_epochs,
    device,
    model_path,
    model_file="model.pt",
    train_log_freq=15,
):

    best_eval_metric = -1
    for epoch in range(n_epochs):
        t1 = time.time()
        for i, (input_nodes, pos_graph, neg_graph, subgraphs) in enumerate(
            train_dataloader
        ):
            batch_features = {
                ntype: {
                    feat_name: feat[node_idx].to(device)
                    for feat_name, feat in features[ntype].items()
                }
                for ntype, node_idx in input_nodes.items()
            }

            subgraphs = [subgraph.to(device) for subgraph in subgraphs]
            pos_scores, neg_scores = model(
                subgraphs, batch_features, pos_graph.to(device), neg_graph.to(device)
            )
            pos_scores, neg_scores = torch.cat(list(pos_scores.values())), torch.cat(
                list(neg_scores.values())
            )

            predictions = torch.cat([pos_scores, neg_scores])
            labels = torch.cat(
                [torch.ones_like(pos_scores), torch.zeros_like(neg_scores)]
            )
            loss = loss_fn(predictions, labels)

            if (i + 1) % train_log_freq == 0:
                print("Train Loss: {:.4f}".format(loss.item()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(
            "Epoch {:05d}:{:05d} | Epoch Time(s) {:.4f}".format(
                epoch + 1, n_epochs, time.time() - t1
            )
        )

        metric, val_loss = evaluate(
            model,
            features,
            loss_fn,
            validation_dataloader,
            eval_edge_mask,
            device=device,
        )
        if metric > best_eval_metric:
            print("Validation average MRR  : {} ".format(metric))
            print("Validation loss : {} ".format(val_loss))
            model.save(os.path.join(model_path, model_file))
            save_eval_metrics({"mrr": metric}, model_path)
            best_eval_metric = metric


def train(data_path, model_path, devices, hyperparameters):
    print(
        "Training config: data_path: {}, model_path: {}, devices: {} hyperparameters: {}".format(
            data_path, model_path, devices, hyperparameters
        )
    )
    device_type = get_device_type(devices)

    graphloader = GraphLoader(data_path)
    g = graphloader.graph

    print("Loaded graph: {}".format(g))
    train_g = dgl.edge_subgraph(g, get_train_eids(g), preserve_nodes=True)
    print("Train graph: {}".format(train_g))
    sampler = dgl.dataloading.MultiLayerNeighborSampler(
        [args.num_neighbors] * args.num_encoder_layers
    )
    negative_sampler = dgl.dataloading.negative_sampler.Uniform(args.num_negs)
    train_dataloader = dgl.dataloading.EdgeDataLoader(
        train_g,
        {
            etype: train_g.edges(etype=etype, form="eid")
            for etype in train_g.canonical_etypes
        },
        sampler,
        negative_sampler=negative_sampler,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_sampler = dgl.dataloading.MultiLayerNeighborSampler(
        [args.num_neighbors] * args.num_encoder_layers
    )
    val_neg_sampler = dgl.dataloading.negative_sampler.Uniform(args.num_negs)
    val_dataloader = dgl.dataloading.EdgeDataLoader(
        g,
        get_valid_eids(g),
        val_sampler,
        negative_sampler=val_neg_sampler,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    eval_edge_mask = get_all_forward_edges(g)

    features_dict = graphloader.get_node_features()

    ### New stuff to handle protein sequence embeddings stored as strings
    if "protT5" in features_dict["protein"]:
        rng = np.random.default_rng()
        features_dict["protein"]["protT5"] = torch.as_tensor(
            [
                (
                    [float(e_str_list) for e_str_list in e_str.split(",")]
                    if e_str != ""
                    else rng.random(1024)
                )
                for e_str in features_dict["protein"]["protT5"]
            ]
        ).float()
    if "esm2" in features_dict["protein"]:
        rng = np.random.default_rng()
        features_dict["protein"]["esm2"] = torch.as_tensor(
            [
                (
                    [float(e_str_list) for e_str_list in e_str.split(",")]
                    if e_str != ""
                    else rng.random(1024)
                )
                for e_str in features_dict["protein"]["esm2"]
            ]
        ).float()
    ###

    input_sizes = {
        ntype: {feat_name: features[feat_name].shape[1] for feat_name in features}
        for ntype, features in features_dict.items()
    }
    print("Got input features with shape graph: {}".format(input_sizes))

    loss_fn = F.binary_cross_entropy_with_logits

    model = get_model(g, input_sizes, hyperparameters, device=device_type)
    print("Created model: {}".format(model))

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=hyperparameters["lr"],
        weight_decay=hyperparameters["weight_decay"],
    )

    print("Starting model training")

    train_n_epochs(
        model,
        optimizer,
        features_dict,
        loss_fn,
        train_dataloader,
        val_dataloader,
        eval_edge_mask,
        hyperparameters["n_epochs"],
        device_type,
        model_path,
    )


def transform(data_path, model_path, devices, hyperparameters):
    hyperparameters = normalize_hyperparameter_keys(hyperparameters)
    print(
        "Transform config: data_path: {}, model_path: {}, devices: {} hyperparameters: {}".format(
            data_path, model_path, devices, hyperparameters
        )
    )
    device_type = get_device_type(devices)

    graphloader = GraphLoader(data_path)
    g = graphloader.graph

    print("Loaded graph: {}".format(g))

    features_dict = graphloader.get_node_features()
    input_sizes = {
        ntype: {feat_name: features[feat_name].shape[1] for feat_name in features}
        for ntype, features in features_dict.items()
    }
    print("Got input features with shape graph: {}".format(input_sizes))

    model = get_model(
        g,
        input_sizes,
        hyperparameters,
        device=device_type,
        model_file=os.path.join(model_path, "model.pt"),
    )
    print("Created model with saved parameters: {}".format(model))

    print("Getting model embeddings")
    node_embeddings = model.get_embeddings(
        g,
        features_dict,
        batch_size=hyperparameters["batch_size"],
        device=device_type,
        num_workers=0,
    )
    relation_type_embeddings = model.decoder.w_relation

    print("Saving model artifacts")
    save_link_prediction_model_artifacts(
        g,
        model_path,
        graphloader,
        hyperparameters,
        node_embeddings,
        relation_type_embeddings,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local",
        action="store_true",
        default=False,
        help="Whether script is running locally",
    )
    parser.add_argument("--name", type=str, default="rgcn-link-predict")
    parser.add_argument("--model", type=str, default="custom")
    parser.add_argument("--task", type=str, default="link_predict")
    parser.add_argument("--num-neighbors", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-negs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--n-epochs", type=int, default=2)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-bases", type=int, default=2)
    parser.add_argument("--num-encoder-layers", type=int, default=2)
    parser.add_argument("--local-data", type=str, default="./data")
    parser.add_argument("--local-model", type=str, default="./output")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.local:
        data_path, model_path, devices = args.local_data, args.local_model, [0]
    else:
        data_path, model_path, devices = get_training_config()

    train(data_path, model_path, devices, vars(args))
