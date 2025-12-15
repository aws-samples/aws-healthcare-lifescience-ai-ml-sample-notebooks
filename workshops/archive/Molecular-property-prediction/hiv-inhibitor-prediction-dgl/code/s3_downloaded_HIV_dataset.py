from dgllife.data.csv_dataset import MoleculeCSVDataset
from dgllife.utils import smiles_to_bigraph
from functools import partial
import pandas as pd

class S3DownloadedHIVDataset(MoleculeCSVDataset):

    def __init__(self, s3downloaded_dir,
                 node_featurizer=None,
                 edge_featurizer=None,
                 log_every=1000,
                 n_jobs=1, mode='sm'):

        smiles_to_graph = partial(smiles_to_bigraph, add_self_loop=True)
        load = False
        cache_file_path = "./none.bin"

        df = pd.read_csv(s3downloaded_dir + "/full.csv")

        ### Check below if we are to ru nthe sagemaker
        #if(mode == 'local'):
        #    df = df.drop(columns=['activity'])

        super(S3DownloadedHIVDataset, self).__init__(df=df,
                                smiles_to_graph=smiles_to_graph,
                                node_featurizer=node_featurizer,
                                edge_featurizer=edge_featurizer,
                                smiles_column='smiles',
                                cache_file_path=cache_file_path,
                                load=load,
                                log_every=log_every,
                                init_mask=True,
                                n_jobs=n_jobs)

    def __getitem__(self, item):
        """Get datapoint with index

        Parameters
        ----------
        item : int
            Datapoint index

        Returns
        -------
        str
            SMILES for the ith datapoint
        DGLGraph
            DGLGraph for the ith datapoint
        Tensor of dtype float32 and shape (T)
            Labels of the ith datapoint for all tasks. T for the number of tasks.
        Tensor of dtype float32 and shape (T)
            Binary masks of the ith datapoint indicating the existence of labels for all tasks.
        str, optional
            Raw screening result, which can be CI, CA, or CM.
        """
        return self.smiles[item], self.graphs[item], self.labels[item], self.mask[item]