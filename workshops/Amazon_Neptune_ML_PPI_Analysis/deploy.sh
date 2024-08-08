#!/bin/bash

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#

############################################################
# Deploy the AWS Batch Architecture for Protein Folding and Design in your AWS account
## Options
# -b S3 bucket name to use for deployment staging
# -s CloudFormation stack name
# -r Deployment region
# -v ID of VPC to use. If left empty, a new VPC will be created.
# -w ID of first private subnet to use.
# -x ID of second private subnet to use.
# -y ID of public subnet to use.
# -z ID of default security group to use.
# -n Instance type for SageMaker notebook instance
#
# Example CMD
# ./deploy.sh \
#   -b "my-deployment-bucket" \
#   -s "my-neptune-ml-stack" \
#   -r "us-east-1" \
#   -v "vpc-12345678" \
#   -w "subnet-12345678" \
#   -x "subnet-12345678" \
#   -y "subnet-12345678" \
#   -z "sg-12345678" \
#   -n "ml.g5.2xlarge"

set -e
unset -v BUCKET_NAME STACK_NAME REGION VPC PRIVATESUBNET1 PRIVATESUBNET2 PUBLICSUBNET \
    DEFAULT_SECURITY_GROUP NOTEBOOK_INSTANCE_TYPE
TIMESTAMP=$(date +%s)

while getopts 'b:s:r:v:w:x:y:z:n:' OPTION; do
    case "$OPTION" in
    b) BUCKET_NAME="$OPTARG" ;;
    s) STACK_NAME="$OPTARG" ;;
    r) REGION="$OPTARG" ;;
    v) VPC="$OPTARG" ;;
    w) PRIVATESUBNET1="$OPTARG" ;;
    x) PRIVATESUBNET2="$OPTARG" ;;
    y) PUBLICSUBNET="$OPTARG" ;;
    z) DEFAULT_SECURITY_GROUP="$OPTARG" ;;
    n) NOTEBOOK_INSTANCE_TYPE="$OPTARG" ;;
    *) exit 1 ;;
    esac
done

[ -z "$STACK_NAME" ] && { STACK_NAME="neptune-ppi"; }
[ -z "$REGION" ] && { INPUT_FILE="us-east-1"; }
[ -z "$VPC" ] && { VPC=""; }
[ -z "$PRIVATESUBNET1" ] && { PRIVATESUBNET1=""; }
[ -z "$PRIVATESUBNET2" ] && { PRIVATESUBNET2=""; }
[ -z "$PUBLICSUBNET" ] && { PUBLICSUBNET=""; }
[ -z "$DEFAULT_SECURITY_GROUP" ] && { DEFAULT_SECURITY_GROUP=""; }
[ -z "$NOTEBOOK_INSTANCE_TYPE" ] && { NOTEBOOK_INSTANCE_TYPE=""; }

zip -r code.zip * -x .\*/\*
aws s3 cp code.zip s3://$BUCKET_NAME/main/code.zip
rm code.zip
echo $BUCKET_NAME
echo $STACK_NAME
echo $REGION
echo $VPC
echo $PRIVATESUBNET1
echo $PRIVATESUBNET2
echo $PUBLICSUBNET
echo $DEFAULT_SECURITY_GROUP
echo $NOTEBOOK_INSTANCE_TYPE
aws cloudformation package --template-file cfn/neptune-ml-nested-stack.json --output-template cfn/neptune-ml-nested-stack-packaged.yaml \
    --region $REGION --s3-bucket $BUCKET_NAME --s3-prefix cfn
aws cloudformation deploy --template-file cfn/neptune-ml-nested-stack-packaged.yaml --capabilities CAPABILITY_IAM --stack-name $STACK_NAME \
    --region $REGION --parameter-overrides S3Bucket=$BUCKET_NAME DBClusterId=$STACK_NAME-neptune \
    VPC=$VPC PrivateSubnet1=$PRIVATESUBNET1 PrivateSubnet2=$PRIVATESUBNET2 PublicSubnet=$PUBLICSUBNET \
    DefaultSecurityGroup=$DEFAULT_SECURITY_GROUP NotebookInstanceType=$NOTEBOOK_INSTANCE_TYPE Timestamp=$TIMESTAMP CodeRepoS3BucketName=$BUCKET_NAME
rm cfn/neptune-ml-nested-stack-packaged.yaml
