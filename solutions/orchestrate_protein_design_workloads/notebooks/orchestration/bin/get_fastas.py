import boto3
import re
import sys
boto_session = boto3.session.Session()
s3 = boto_session.client("s3")

s3_uri=sys.argv[1]
def get_files_within_s3_uri(s3_uri):
       
        mybucket=re.findall("s3://(.*?)\/",s3_uri)[0]
        myprefix=re.findall("s3://.*?\/(.*)",s3_uri)[0]
        object_list = []
        object_list_2=[]
        try:
            paginator = s3.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=mybucket, Prefix=myprefix)

            for page in pages:
                for obj in page['Contents']:
                    object_list.append(obj['Key'].rstrip())

            object_list_2 = [f's3://{mybucket}/{_}' for _ in object_list if _.endswith('.fas')]

            return (object_list_2)
        except Exception as e:
            print(e)
my_objects=get_files_within_s3_uri(s3_uri)
#for i in range(0,len(my_objects)):
#    print(f'''{i} {my_objects[i]}''',end="\n")
print(*my_objects,sep="\n")
