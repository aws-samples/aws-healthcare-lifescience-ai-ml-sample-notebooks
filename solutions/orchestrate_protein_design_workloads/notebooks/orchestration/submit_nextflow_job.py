import boto3
import os
import random
import sagemaker
sess = sagemaker.Session()
bucket = sess.default_bucket()   
random_str=str(random.randint(100000, 9999999))
#modify per your orchestrator  compute environment and nextflow job definition from the Batch console
orchestrator_compute_environment="CPUOnDemandJobQueue-Wn5WSyuTU2sZehux" 
nextflow_job_definition="NextflowJobDefinition-894c4271a53b004"

nextflow_script="run_rfdesign_esmfold_multiple_sequences.nf"

my_asset_uri=f"s3://{bucket}/assets_input" #modify to your own bucket
my_input_bucket=f"s3://{bucket}/pd1-demo/"
rf_design_output=f"s3://{bucket}/myrfdesign_hallucination_{random_str}"
esmf_output=f"s3://{bucket}/FinalESMFoldOutput_{random_str}/"
print(my_asset_uri)
print(rf_design_output)
print(esmf_output)

#move input files and code to their respective buckets in S3.
#copy pdb structures
os.system(f'aws s3 cp  pd1_demo/pd1.pdb s3://{bucket}/pd1-demo/') 
os.system(f'aws s3 cp  pd1_demo/pdl1.pdb s3://{bucket}/pd1-demo/') 

#copy dependencies to s3 in the bin directory
os.system(f'aws s3 cp --recursive bin/ {my_asset_uri}/bin/') 
os.system(f'aws s3 cp {nextflow_script} {my_asset_uri}/') #copy nextflow script to s3

#Next we specify the commands for the nextflow orchestrator to run.
#First we copy in the data from the asset bucket, which includes the .nf script and dependencies
#Next we run the .nf script, and print a finished message when done.
nextflow_commands=[
    f'''aws s3 cp --recursive {my_asset_uri} . 
    nextflow run {nextflow_script} --s3_input {my_input_bucket}  --rf_design_output  {rf_design_output} --esmfold_output {esmf_output};
    echo Finished'''
]
    
client = boto3.client('batch')
response = client.submit_job(
    jobName=f'nextflow_job_{random_str}',
    jobQueue=orchestrator_compute_environment, #modify this to your own JobQueue
    jobDefinition=nextflow_job_definition, #modify this to your own Job Definition
    containerOverrides={'command':nextflow_commands}
)
print(response)
