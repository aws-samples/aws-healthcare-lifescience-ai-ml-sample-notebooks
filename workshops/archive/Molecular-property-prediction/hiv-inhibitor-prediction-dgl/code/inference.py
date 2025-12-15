import logging
import sys
import json
from argparse import Namespace
from dgllife.utils import smiles_to_bigraph
from functools import partial

import pandas as pd

from utils import load_model, init_featurizers, model_saved_path, model_params_saved_path
import torch
from dgllife.data.csv_dataset import MoleculeCSVDataset

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

node_featurizer = None

def _set_node_featurizer():
    global node_featurizer
    node_featurizer = init_featurizers('canonical')[0]

def model_fn(model_dir):

    logger.info('model_fn')
    logger.info('Loading the trained model...')

    model_params_path = model_params_saved_path(model_dir)
    model_path = model_saved_path(model_dir)
    logger.info("Model is loaded from path [{}]".format(model_params_path))

    with open(model_params_path, 'rb') as f:
        model_params = json.load(f)

    args = Namespace(**model_params)
    logger.info("Model parameters is loaded from path [{}]".format(args))

    global node_featurizer
    node_featurizer = init_featurizers(args.gnn_featurizer_type)[0]
    model = load_model(args, node_featurizer)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f, map_location=device)["model_state_dict"])

    logger.info("Model loaded.")
    return model.to(device)

def input_fn(request_body, request_content_type):

    if request_content_type == "application/json":
        logger.debug("Request body is : [{}]".format(request_body))
        smiles = json.loads(request_body)['smiles']
        logger.debug("Input molecule smiles are : [{}]".format(smiles))
        smiles_to_graph = partial(smiles_to_bigraph, add_self_loop=True)
        graphs = []

        global node_featurizer
        logger.debug("Converting smiles to graphs using function [{}]. Node featurizer [{}]".format("smiles_to_graph", node_featurizer))
        for s in smiles:
            logger.debug('Processing molecule {}'.format(s))
            graph_data = smiles_to_graph(s, node_featurizer=node_featurizer,
                            edge_featurizer=None)
            graphs.append(graph_data)
        return graphs

    else:
        raise ValueError("Unsupported content type: {}".format(request_content_type))

def predict_fn(input_data, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    with torch.no_grad():
        predictions = []
        for graph in input_data:
            node_feats = graph.ndata.pop('h').to(device)
            output = model(graph, node_feats)
            #_, prediction = torch.max(output, dim=1)
            predictions.append(output[0][0])
        return predictions

# if __name__ == '__main__':
#     model = model_fn("/Users/sariyawa/myworkspace/non-backup/source/hiv-inhibitor-prediction-dgl/generated/model/")

#     smiles_json = '{ \
#         "smiles" : [ \
#             "CC(C)(CCC(=O)O)CCC(=O)O" \
#         ] \
#     }'
#     graphs = input_fn(smiles_json, "application/json")
#     predict_fn(graphs, model)


