#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Check if a commit message argument is provided
if [ -z "$1" ]; then
  echo "Error: No commit message provided."
  echo "Usage: ./deploy_jekyll.sh 'Your commit message'"
  exit 1
fi

# Build the Jekyll site
echo "Building Jekyll site..."
bundle exec jekyll build

# Remove the old contents of the docs directory
echo "Removing old documentation..."
rm -rf ../aws-healthcare-lifescience-ai-ml-sample-notebooks/docs/*

# Copy the newly built site to the docs directory
echo "Copying new site to docs directory..."
cp -r _site/* ../aws-healthcare-lifescience-ai-ml-sample-notebooks/docs/

# Change to the aws-healthcare-lifescience-ai-ml-sample-notebooks directory
echo "Changing to aws-healthcare-lifescience-ai-ml-sample-notebooks directory..."
cd ../aws-healthcare-lifescience-ai-ml-sample-notebooks

# Stage the docs directory for Git
echo "Staging changes in docs/ for Git..."
git add docs/

# Commit with the provided commit message
echo "Committing changes..."
git commit -m "$1"

# Push the changes to the remote repository
echo "Pushing changes to the remote repository..."
git push

echo "Done."
