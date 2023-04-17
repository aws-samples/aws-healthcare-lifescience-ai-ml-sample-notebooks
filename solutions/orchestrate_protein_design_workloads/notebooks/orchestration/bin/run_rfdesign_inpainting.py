# Import required Python packages
import boto3
from batchfold.batchfold_environment import BatchFoldEnvironment
from batchfold.rfdesign_job import RFDesignHallucinateJob, RFDesignInpaintJob
from batchfold.utils import utils
from Bio.PDB import PDBParser, PDBIO, Selection
from Bio.PDB.PDBList import PDBList
from datetime import datetime
from IPython import display
#import matplotlib.pyplot as plt
import os
import numpy as np
import py3Dmol

import logging
import sys
import argparse
#logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.basicConfig(stream=sys.stderr, level=logging.INFO)

parser = argparse.ArgumentParser(description='Parse the options')
parser.add_argument('--input_s3_uri', dest='input_s3_uri', default=None,
                   help='input s3 uri (default: None)')
parser.add_argument('--output_s3_uri', dest='output_s3_uri', default=None,
                   help='output_s3_uri (default: None)')
parser.add_argument('--num_sequences_to_generate', dest='num_sequences_to_generate', default=1,
                   help='number of sequences for rfdesign to generate (default: 1)')


args = parser.parse_args()
args=vars(args)

input_s3_uri=args['input_s3_uri']
output_s3_uri=args['output_s3_uri']
num_sequences_to_generate=args['num_sequences_to_generate']


# Create AWS clients
boto_session = boto3.session.Session()
s3 = boto_session.client("s3")
batch_environment = BatchFoldEnvironment(boto_session=boto_session)

total_num = num_sequences_to_generate
hallucinate_job_prefix = "RFDesignHallucinateJob" + datetime.now().strftime("%Y%m%d%s")
job_queue_name = "G4dnJobQueue"

inpainting_job_name = "RFDesignInpaintingJob" + datetime.now().strftime("%Y%m%d%s")
job_queue_name = "G4dnJobQueue"
params = {
    "contigs":"25-35,B63-82,15-25,B119-140,0-15",
    "len": "80-115",
    "num_designs": total_num,
    "dump_all": True,
}
new_job = RFDesignInpaintJob(
    boto_session=boto_session,
    job_name = inpainting_job_name,
    target_id = "4ZQK",
    input_s3_uri = input_s3_uri,
    output_s3_uri = output_s3_uri,
    pdb = "input/pd1.pdb",
    params = params
)

#print(f"Submitting {job_name}")
#print(new_job)
submission = batch_environment.submit_job(new_job, job_queue_name)
print (submission.job_id)
