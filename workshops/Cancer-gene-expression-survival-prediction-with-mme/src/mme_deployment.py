import boto3
import os

sm_client = boto3.client("sagemaker")
s3_client = boto3.client("s3")
s3_resource = boto3.resource("s3")

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def model_from_registry(model_package_arn):
    
    response = sm_client.describe_model_package(
        ModelPackageName=model_package_arn
    )
    
    model_data_url = response["InferenceSpecification"]["Containers"][0]["ModelDataUrl"]
    
    return model_data_url

    
def deploy_to_mme_location(model_data_url, mme_model_location_s3, genome_group):
    
    print("Deploying models from [{}] to [{}]".format(model_data_url, mme_model_location_s3))
    
    _, path = mme_model_location_s3.split(":", 1)
    path = path.lstrip("/")
    bucket, path = path.split("/", 1)
    
    _, path_source = model_data_url.split(":", 1)
    source = path_source.lstrip("/")
    
    response = s3_client.copy_object(Bucket = bucket, CopySource = source, Key=path + "/model-{}.tar.gz".format(genome_group))
    
    print(response)
    

    
if __name__ == "__main__":
    
    model_package_arn = os.environ['modelPackageArn']
    mme_model_location_s3 = os.environ['mmeModelLocation']
    genome_group = os.environ['genomeGroup']
    
    print("Preparing MME the deployment for model package arn [{}].".format(model_package_arn))
    
    model_data_url = model_from_registry(model_package_arn)
    
    print("Model url found. [{}]".format(model_data_url))
    
    deploy_to_mme_location(model_data_url, mme_model_location_s3, genome_group)

    