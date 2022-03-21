from torch.nn import functional as F

def model_saved_path(base):
    return base + "/model.pth"

def model_params_saved_path(base):
    return base + '/model_params.json'

def load_model(args, node_featurizer, n_tasks=1):

    num_gnn_layers = len(args.gnn_hidden_feats)
    model = None
    if(args.gnn_model_name == 'GCN-p'):
        from dgllife.model import GCNPredictor
        model = GCNPredictor(
            in_feats=node_featurizer.feat_size(),
            hidden_feats=args.gnn_hidden_feats,
            activation=[F.relu] * num_gnn_layers,
            residual=[args.gnn_residuals] * num_gnn_layers,
            batchnorm=[args.gnn_batchnorm] * num_gnn_layers,
            dropout=[args.gnn_dropout] * num_gnn_layers,
            predictor_hidden_feats=args.gnn_predictor_hidden_feats,
            predictor_dropout=args.gnn_dropout,
            n_tasks=n_tasks
        )
    elif (args.gnn_model_name == 'GAT-p'):
        from dgllife.model import GATPredictor
        model = GATPredictor(
            in_feats=node_featurizer.feat_size(),
            hidden_feats=args.gnn_hidden_feats,
            num_heads=[args.gnn_num_heads] * num_gnn_layers,
            feat_drops=[args.gnn_dropout] * num_gnn_layers,
            attn_drops=[args.gnn_dropout] * num_gnn_layers,
            alphas=[args.gnn_alphas] * num_gnn_layers,
            residuals=[args.gnn_residuals] * num_gnn_layers,
            predictor_hidden_feats=args.gnn_predictor_hidden_feats,
            predictor_dropout=args.gnn_dropout,
            n_tasks=n_tasks
        )
    return model


def init_featurizers(featurizer_type):
    node_feaurizer = None
    edge_featurizer = None
    if(featurizer_type == 'canonical'):
        from dgllife.utils import CanonicalAtomFeaturizer
        node_feaurizer = CanonicalAtomFeaturizer()
    elif(featurizer_type == 'attentivefp'):
        from dgllife.utils import AttentiveFPAtomFeaturizer
        node_feaurizer = AttentiveFPAtomFeaturizer()
    else:
        raise ValueError(
            "Expect featurizer_type to be in ['canonical', 'attentivefp'], "
            "got {}".format(featurizer_type))
    return node_feaurizer, edge_featurizer