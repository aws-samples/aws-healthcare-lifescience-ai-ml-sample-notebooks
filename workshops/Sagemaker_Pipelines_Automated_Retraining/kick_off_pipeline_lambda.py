import json
import boto3
import logging
import os
import copy


# get environment variables
# name of bucket lambda gets notifications from
NOTIFICATION_BUCKET_NAME = os.environ["NOTIFICATION_BUCKET_NAME"]
SAGEMAKER_PIPELINE_NAME = os.environ["SAGEMAKER_PIPELINE_NAME"]


def read_in_file_from_s3(bucketname, filename):
    """reads in the file from S3 and returns the content from the body of the file"""
    s3 = boto3.resource("s3")
    obj = s3.Object(bucketname, filename)
    body = obj.get()["Body"].read()
    return body


def convert_to_s3uri(bucketname, filename):
    the_uri = f"s3://{bucketname}/{filename}"
    return the_uri


def kick_off_sagemaker_pipeline(pipelinename=None, s3uri=None):
    client = boto3.client("sagemaker")
    PipelineParameters = [
        {"Name": "InputData", "Value": f"{s3uri}"},
    ]
    response = client.start_pipeline_execution(
        PipelineName=pipelinename, PipelineParameters=PipelineParameters
    )
    return response


def lambda_handler(event, context):
    # uncomment to log event info
    # logging.info(json.dumps(event))

    filename = event["Records"][0]["s3"]["object"]["key"]
    filename_basename = os.path.basename(filename)

    the_s3uri = convert_to_s3uri(NOTIFICATION_BUCKET_NAME, filename)
    the_response = kick_off_sagemaker_pipeline(
        pipelinename=SAGEMAKER_PIPELINE_NAME, s3uri=the_s3uri
    )
    # logging.info(json.dumps(content_4))
    # put_file_in_s3(f'''{filename_basename}_out''',json.dumps(content_4),OUTPUT_BUCKET_NAME)

    return {"statusCode": 200, "body": json.dumps("Hello from Lambda!")}
