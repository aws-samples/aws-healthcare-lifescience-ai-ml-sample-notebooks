import boto3
import sys
import time
client = boto3.client('batch')
def wait_until_job_is_done(job_id):
    x=1
    while (x==1):
        response=client.describe_jobs(jobs=[job_id])
        the_status=response['jobs'][0]['status']
        if the_status in ['SUCCEEDED','FAILED']:
            return()
        else:
            time.sleep(10) #wait a bit before checking the status again

jobs_file=sys.argv[1]
jobs_list=open(jobs_file).readlines()
jobs_list=[i.rstrip() for i in jobs_list]

for i in jobs_list:
    wait_until_job_is_done(i)
