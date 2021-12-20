import json
import boto3
import logging
import os
import copy

#get environment variables
#name of bucket lambda gets notifications from
NOTIFICATION_BUCKET_NAME = os.environ['NOTIFICATION_BUCKET_NAME']
OUTPUT_BUCKET_NAME=os.environ['OUTPUT_BUCKET_NAME']

#hard coded list of breast cancer genes


def read_in_file_from_s3(bucketname,filename):
    '''reads in the file from S3 and returns the content from the body of the file'''
    s3 = boto3.resource('s3')
    obj = s3.Object(bucketname, filename)
    body = obj.get()['Body'].read()
    return(body)


def put_file_in_s3(filename,filecontent,output_bucketname):
    '''add file to s3 bucket, return response of operation'''
    s3 = boto3.client('s3')
    response = s3.put_object(
        Bucket=output_bucketname,
        Key=filename,
        Body=filecontent,
    )
    return (response)

def call_textract(bucketname,filename):

    client = boto3.client('textract')
    x=client.detect_document_text(Document={'S3Object':{'Bucket':bucketname,'Name':filename}})
    the_blocks=x['Blocks']
    all_text=''
    for i in range(0,len(the_blocks)):
        try:
            if the_blocks[i]["BlockType"]=='WORD':
                the_text=the_blocks[i]['Text']
                all_text=all_text + " " + the_text
        except:
            pass
    return(all_text)


def call_comprehehend_medical(the_input=None,call_type="detect_entities_v2"):
    '''pass the input data to comprehend medical
    call_type controls what NLP operation comprehend medical should do.
    call_type must be a valid method for CM.
    '''
    structured_content=None
    client = boto3.client('comprehendmedical')
    if call_type=='detect_entities_v2':
        structured_content = client.detect_entities_v2(Text=the_input)
    elif call_type=='infer_icd10_cm':
        structured_content = client.infer_icd10_cm(Text=the_input)
    elif call_type=='infer_rx_norm':
        structured_content = client.infer_rx_norm(Text=the_input)
    else:
        logging.warning(f'Something is Wrong. Comprehend Medical call type {call_type} may be invalid.')
    try:
        response=structured_content['ResponseMetadata']['HTTPStatusCode']
        if response==200:
            pass
    except:
        logging.warning('Something is wrong. Perhaps there is a problem calling Comprehend Medical?')
        structured_content=None

    return(structured_content)


def call_comprehend(the_input=None):
    '''call the custom comprehend model that has been previously trained to classify the document according to its medical specialty type.'''
    client = boto3.client('comprehend')
    response= client.detect_sentiment(Text=the_input,LanguageCode='en')
    #response = client.classify_document(
    #    Text=the_input,
    #    EndpointArn=endpoint_arn
    #    )
    return(response)


def lambda_handler(event, context):
    #uncomment to log event info
    #logging.info(json.dumps(event))

    filename=event['Records'][0]['s3']['object']['key']
    filename_basename=os.path.basename(filename)

    #content=read_in_file_from_s3(NOTIFICATION_BUCKET_NAME,filename)
    content=call_textract(NOTIFICATION_BUCKET_NAME,filename)
    content_2=call_comprehehend_medical(the_input=content,call_type='detect_entities_v2') # .decode("utf-8") decode to prevent error
    custom_predictions=call_comprehend(the_input=content) #.decode("utf-8")
    content_3=copy.deepcopy(content_2) #make copy to avoid modifying original dictionary
    content_3['Comprehend_Detected_Entities']=custom_predictions
    content_3['Raw Text']=content
    content_4=content_3


    #export final output
    #logging.info(json.dumps(content_4))
    put_file_in_s3(f'''{filename_basename}_out''',json.dumps(content_4),OUTPUT_BUCKET_NAME)


    return {
        'statusCode': 200,
        'body': json.dumps('Hello from Lambda!')
    }