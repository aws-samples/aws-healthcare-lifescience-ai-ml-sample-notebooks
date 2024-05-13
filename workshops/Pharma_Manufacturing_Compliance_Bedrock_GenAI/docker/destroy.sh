#!/bin/bash

STACK_NAME=manufacturing-document-comparison
IMAGE=manufacturing_document_comparison

region=us-east-1
export AWS_DEFAULT_REGION=${region}

aws cloudformation delete-stack --stack-name "${STACK_NAME}"

aws ecr delete-repository --repository-name"${IMAGE}" --force > /dev/null 2>&1

echo "Stack ${STACK_NAME} has been destroyed successfully."