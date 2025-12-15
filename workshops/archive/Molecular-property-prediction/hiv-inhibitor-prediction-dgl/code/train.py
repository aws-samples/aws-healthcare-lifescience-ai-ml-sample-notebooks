import argparse
import os
import json

import dgl
import numpy as np
import pandas as pd
import torch
from dgllife.utils import EarlyStopping, Meter
from numpy import double
from torch.optim import Adam
from torch.utils.data import DataLoader
from utils import load_model, init_featurizers, model_saved_path, model_params_saved_path
from s3_downloaded_HIV_dataset import S3DownloadedHIVDataset


def run_a_train_epoch(args, epoch, model, data_loader, loss_criterion, optimizer, device):
    model.train()
    train_meter = Meter()
    for batch_id, batch_data in enumerate(data_loader):
        smiles, bg, labels, masks = batch_data
        if len(smiles) == 1:
            # Avoid potential issues with batch normalization
            continue

        labels, masks = labels.to(device), masks.to(device)
        logits = predict(model, bg, device)
        # Mask non-existing labels
        loss = (loss_criterion(logits, labels) * (masks != 0).float()).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_meter.update(logits, labels, masks)
        if batch_id % args.print_every == 0:
            print('epoch [{:d}] of [{:d}], batch {:d}/{:d}, loss [{:.4f}]'.format(
                epoch + 1, args.epochs, batch_id + 1, len(data_loader), loss.item()))
    train_score = np.mean(train_meter.compute_metric(args.metric))
    print('epoch [{:d}] of [{:d}], training:{} [{:.4f}]'.format(
        epoch + 1, args.epochs, args.metric, train_score))


def run_an_eval_epoch(args, model, data_loader, device):
    model.eval()
    eval_meter = Meter()
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            smiles, bg, labels, masks = batch_data
            labels = labels.to(device)
            logits = predict(model, bg, device)
            eval_meter.update(logits, labels, masks)
    return np.mean(eval_meter.compute_metric(args.metric))

def train(args):

    print("Hello I am training with following args.")

    print(args)

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    print("Device : [{}]".format(device))

    node_featurizer, edge_featurizer = init_featurizers(args.gnn_featurizer_type)

    dataset = S3DownloadedHIVDataset(args.full_data,
                                     node_featurizer=node_featurizer,
                                     edge_featurizer=edge_featurizer,
                                     n_jobs=1 if args.num_workers == 0 else args.num_workers, mode=args.mode)

    train_set, val_set = split_dataset(args, dataset)

    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_molgraphs, num_workers=args.num_workers)

    val_loader = DataLoader(dataset=val_set, batch_size=args.batch_size,
                            collate_fn=collate_molgraphs, num_workers=args.num_workers)

    model = load_model(args, node_featurizer).to(device)

    print(model)

    loss_criterion = torch.nn.BCEWithLogitsLoss(reduction='none')

    optimizer = Adam(model.parameters(), lr=args.learning_rate,
                     weight_decay=args.weight_decay)

    stopper = EarlyStopping(patience=args.patience,
                            filename=model_saved_path(args.model_dir),
                            metric=args.metric)

    for epoch in range(args.epochs):

        # Train
        run_a_train_epoch(args, epoch, model, train_loader, loss_criterion, optimizer, device)

        # Validation and early stop
        val_score = run_an_eval_epoch(args, model, val_loader, device)
        early_stop = stopper.step(val_score, model)
        print('epoch [{:d}] of [{:d}], validation:{} [{:.4f}], best validation:{} [{:.4f}]'.format(
            epoch + 1, args.epochs, args.metric,
            val_score, args.metric, stopper.best_score))

        if early_stop:
            break
    save_model_args(args)


def split_dataset(args, dataset):

    train_set = dgl.data.utils.Subset(dataset, pd.read_csv(args.train_data + "/train.csv").indices.to_numpy())
    val_set = dgl.data.utils.Subset(dataset, pd.read_csv(args.val_data + "/validation.csv").indices.to_numpy())

    return train_set, val_set


def predict(model, bg, device):

    bg = bg.to(device)

    node_feats = bg.ndata.pop('h').to(device)
    #edge_feats = bg.edata.pop('e').to(args['device'])
    return model(bg, node_feats)


def collate_molgraphs(data):
    """Batching a list of datapoints for dataloader.
    Parameters
    ----------
    data : list of 4-tuples.
        Each tuple is for a single datapoint, consisting of
        a SMILES, a DGLGraph, all-task labels and optionally a binary
        mask indicating the existence of labels.
    Returns
    -------
    smiles : list
        List of smiles
    bg : DGLGraph
        The batched DGLGraph.
    labels : Tensor of dtype float32 and shape (B, T)
        Batched datapoint labels. B is len(data) and
        T is the number of total tasks.
    masks : Tensor of dtype float32 and shape (B, T)
        Batched datapoint binary mask, indicating the
        existence of labels.
    """

    smiles, graphs, labels, masks = map(list, zip(*data))

    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    labels = torch.stack(labels, dim=0)
    masks = torch.stack(masks, dim=0)

    return smiles, bg, labels, masks


def save_model_args(args):
    filename = model_params_saved_path(args.model_dir)
    gnn_params_keys = filter(lambda arg :  arg.startswith("gnn"), vars(args))
    gnn_params = dict(map(lambda key : (key, args.__dict__[key]), gnn_params_keys))
    file = open(filename, "w")
    json.dump(gnn_params, file)
    file.close()


if __name__ == "__main__":
    print("Loading Parameters\n")
    parser = argparse.ArgumentParser('HIV Inhibitor Binary Classification')


    # Feature engineering hyper-params
    parser.add_argument('-f', '--gnn-featurizer-type', choices=['canonical', 'attentivefp'],
                        default='canonical',
                        help='Featurization for atoms (and bonds). This is required for models '
                             'other than gin_supervised_**.')

    # model evaluation hyper-params
    parser.add_argument('-me', '--metric', choices=['roc_auc_score', 'pr_auc_score'],
                        default='roc_auc_score',
                        help='Metric for evaluation (default: roc_auc_score)')

    # model architecture hyper-params

    parser.add_argument('-nw', '--num-workers', type=int, default=1,
                        help='Number of processes for data loading (default: 1)')

    parser.add_argument('-mn', '--gnn-model-name', choices=['GCN-p', 'GAT-p'], default='GCN-p',
                        help='DGL Life model implementation to be used. ')

    parser.add_argument('-hl', '--gnn-hidden-feats', type=int, default=[256],
                        help='No of hidden GCNLayers to to be use i.e hoe many nerighers to be considered.')

    parser.add_argument('-res', '--gnn-residuals', type=bool, default=False,
                        help='Whether to use residual connections in the GCNLayer or not.')

    parser.add_argument('-batchnorm', '--gnn-batchnorm', type=bool, default=True,
                        help='Whether to use batch norm in each GCNLayer or not.')

    parser.add_argument('-dropout', '--gnn-dropout', type=double, default=0.001,
                        help='Drop out percentage')

    parser.add_argument('-al', '--gnn-alphas', type=double, default=0.08,
                        help='Alphas')

    parser.add_argument('-nh', '--gnn-num-heads', type=double, default=8,
                        help='Number of heads')

    parser.add_argument('-predictor_hidden_feats', '--gnn-predictor-hidden-feats', type=int, default=512,
                        help='')

    # Training hyper-params

    parser.add_argument('-bs', '--batch-size', type=int, default=512,
                        help='Batch size for the data loaders (default : 32)')

    parser.add_argument('--epochs', type=int, default=1000,
                        help='Maximum number of epochs for training. '
                             'We set a large number by default as early stopping '
                             'will be performed. (default: 3)')

    parser.add_argument('-lr', '--learning-rate', type=double, default=0.001,
                        help='Learning rate')

    parser.add_argument('-wd', '--weight-decay', type=double, default=0.001,
                        help='Weight decay.')

    parser.add_argument('-patience', '--patience', type=int, default=30,
                        help='')

    # Monitoring params
    parser.add_argument('-pe', '--print-every', type=int, default=20,
                        help='Print the training progress every X mini-batches')

    parser.add_argument('-md', '--mode', type=str, default="local",
                        help='Mode of running this script [sm, local]')


    # Container environment
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--full-data", type=str, default=os.environ["SM_CHANNEL_DATA_FULL"])
    parser.add_argument("--train-data", type=str, default=os.environ["SM_CHANNEL_DATA_TRAIN"])
    parser.add_argument("--val-data", type=str, default=os.environ["SM_CHANNEL_DATA_VAL"])

    #parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])

    args = parser.parse_args()
    train(args)

