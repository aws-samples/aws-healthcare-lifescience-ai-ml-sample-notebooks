# Manufacturing Document Comparison

## Description
This project automates the deployment of a manufacturing document comparison application on AWS. It builds and pushes a Docker image to Amazon ECR and then uses CloudFormation to provision the required resources.

## Prerequisites
- AWS CLI, configured for your account.
- Docker.
- A CloudFormation template named cf.yaml.

## Getting Started
Clone the Repository

```bash
git clone <repository-url>
cd <repository-directory>
```
## Configuration
Adjust AWS CLI and Docker if not already set up. The script defaults to the us-east-1 region.

## Build and Deploy
Execute the script to build the Docker image, push it to ECR, and deploy your CloudFormation stack:

```bash
./deploy.sh
```

## Destroy
```bash
./destroy.sh
```

## What the Script Does
- Checks for the ECR repository; creates it if absent.
- Logs into ECR.
- Builds and tags the Docker image.
- Pushes the image to ECR.
- Deploys/updates the CloudFormation stack with the image URI.
- Customize the Script

## Modify these variables in deploy.sh as needed:
- **IMAGE**: Docker image and ECR repository name.
- **SERVICE_NAME**: Service name for CloudFormation.
- **STACK_NAME**: CloudFormation stack name.

## Additional Information
Ensure cf.yaml is in the same directory as deploy.sh, or update the script with the correct path.

## Troubleshooting
Check AWS CLI credentials and CloudFormation console for errors.

# License
Apache-2.0 License