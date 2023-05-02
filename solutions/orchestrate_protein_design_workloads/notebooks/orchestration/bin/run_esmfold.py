import boto3
from datetime import datetime
import matplotlib.pyplot as plt
from batchfold.batchfold_environment import BatchFoldEnvironment
from batchfold.batchfold_target import BatchFoldTarget
from batchfold.jackhmmer_job import JackhmmerJob
from batchfold.openfold_job import OpenFoldJob
from batchfold.alphafold2_job import AlphaFold2Job
from batchfold.omegafold_job import OmegaFoldJob
from batchfold.esmfold_job import ESMFoldJob
from batchfold.utils import utils
#from IPython import display
import numpy as np
import os
import sys
input_s3_uri=sys.argv[1]
output_s3_uri_base=sys.argv[2]
#put the new file in a new directory in s3 based on previous output name
output_s3_uri=output_s3_uri_base+os.path.basename(input_s3_uri).removesuffix(".fas")

# Create AWS clients
boto_session = boto3.session.Session() #add profile_name="A_PROFILE" if desired

batch_environment = BatchFoldEnvironment(boto_session=boto_session)


my_datetime=datetime.now().strftime("%Y%m%d%s")
job_name = "jb_target" + "_ESMFoldJob_" + my_datetime
esmfold_job = ESMFoldJob(
    job_name=job_name,
    target_id="my_target",
    fasta_s3_uri=input_s3_uri,
    output_s3_uri=output_s3_uri,
    boto_session=boto_session,
    cpu=8,
    memory=31,  # Why not 32? ECS needs about 1 GB for container services
    gpu=1,
)
print(esmfold_job)
esmfold_submission = batch_environment.submit_job(
    esmfold_job, job_queue_name="G4dnJobQueue"
)
