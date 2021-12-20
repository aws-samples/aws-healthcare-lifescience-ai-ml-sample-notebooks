  
#this script installs the relevant dependencies for the lambda function locally and zips it with the lambda function
rm -rf package # remove the package directory if it exists already
cat requirements.txt |while read line; do pip install --target ./package $line; done
cd package/;zip -r9 ${OLDPWD}/lambda.zip .;cd ..;zip -g lambda.zip  ai_ml_services_lambda.py;
