#!/bin/bash

IMAGE=manufacturing_document_comparison

SERVICE_NAME=manufacturing_document_comparison
STACK_NAME=manufacturing-document-comparison

region=us-east-1
account=$(aws sts get-caller-identity --query Account --output text)
export AWS_DEFAULT_REGION=${region}

fullname="${account}.dkr.ecr.${region}.amazonaws.com/${IMAGE}:latest"

# If the repository doesn't exist in ECR, create it.

aws ecr describe-repositories --repository-names "${IMAGE}" > /dev/null 2>&1

if [ $? -ne 0 ]
then
    aws ecr create-repository --repository-name "${IMAGE}" > /dev/null
fi

# Get the login command from ECR and execute it directly
aws ecr get-login-password --region "${region}" | docker login --username AWS --password-stdin "${account}".dkr.ecr."${region}".amazonaws.com

# Build the docker IMAGE locally with the IMAGE name and then push it to ECR
# with the full name.

docker build -t ${IMAGE} .
docker image tag ${IMAGE} ${fullname}

docker push ${fullname}

# Deploy the CloudFormation stack (create or update as necessary) and suppress the output
aws cloudformation deploy \
    --template-file cf.yaml \
    --stack-name "${STACK_NAME}" \
    --parameter-overrides ServiceName="${SERVICE_NAME}" ImageUri="${fullname}" \
    --capabilities CAPABILITY_NAMED_IAM > /dev/null

# Check if the stack deploy command was successful
if [ $? -eq 0 ]; then
    echo "Stack ${STACK_NAME} has been created or updated successfully."
else
    echo "Error deploying stack ${STACK_NAME}."
fi