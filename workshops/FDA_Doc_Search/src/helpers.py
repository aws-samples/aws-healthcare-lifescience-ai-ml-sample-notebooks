from bs4 import BeautifulSoup
import boto3
from datetime import datetime
import json
import os
import re
import requests

def copy_url_to_s3(
    source_uri, dest_s3_bucket, dest_s3_key, metadata=None, session=boto3.Session()
):
    response = requests.get(source_uri, stream=True, timeout=5)
    s3 = session.resource("s3")
    bucket = s3.Bucket(dest_s3_bucket)
    if metadata is not None:
        bucket.upload_fileobj(
            response.raw, dest_s3_key, ExtraArgs={"Metadata": metadata}
        )
    else:
        bucket.upload_fileobj(response.raw, dest_s3_key)

    return os.path.join("s3://", dest_s3_bucket, dest_s3_key)


def parse_cfm(url):
    parent = os.path.split(url)[0]
    text = requests.get(url, timeout=5).text
    soup = BeautifulSoup(text, features="lxml")
    child_docs = []
    for li in soup.find_all("li"):
        href = str(li.a.attrs.get("href"))
        if re.match("(?!http).*\.[a-z]{3}$", href):
            child_docs.append(
                {"name": href, "url": os.path.join(parent, href), "title": li.a.string}
            )
    return child_docs


def parse_fda_doc_info(doc):
    output = {
        "date": doc.get("date"),
        "extension": os.path.splitext(doc.get("url"))[1],
        "id": doc.get("id"),
        "name": os.path.basename(doc.get("url")),
        "title": doc.get("title", doc.get("type")),
        "url": doc.get("url"),
    }
    return output


def get_kendra_doc_content_type(extension):
    match extension:
        case ".pdf":
            return "PDF"
        case ".html":
            return "HTML"
        case ".xml":
            return "XML"
        case ".xslt":
            return "XSLT"
        case ".md":
            return "MD"
        case ".csv":
            return "CSV"
        case ".xls":
            return "XLS"
        case ".xlsx":
            return "XLSX"
        case ".json":
            return "JSON"
        case ".rtf":
            return "RTF"
        case ".ppt":
            return "PPT"
        case ".docx":
            return "DOCX"
        case ".txt":
            return "TXT"
        case _:
            return "TXT"


def create_doc_metadata(
    result, submission, doc_info, s3_bucket, s3_key, parent_doc_info=None
):

    date = doc_info.get("date") or parent_doc_info.get("date") or ""
    metadata = {
        "Title": doc_info.get("title"),
        "ContentType": get_kendra_doc_content_type(doc_info.get("extension")),
        "Attributes": {
            "_category": "drugs@fda",
            "ApplicationNumber": result.get("application_number", ""),
            "BrandName": result.get("openfda", {}).get("brand_name", [""])[0],
            "GenericName": result.get("openfda", {}).get("generic_name", [""])[0],
            "ManufacturerName": result.get("openfda", {}).get(
                "manufacturer_name", [""]
            )[0],
            "DocumentId": os.path.join("s3://", s3_bucket, s3_key),
            "Submission": f"{submission.get('submission_type')}-{submission.get('submission_number')}",
            "_source_uri": doc_info.get("url", ""),
            "_created_at": datetime.strptime(date, "%Y%m%d").astimezone().isoformat(),
        },
    }
    return json.dumps(metadata)


def write_string_to_s3(string, bucket, key, session=boto3.Session()):
    s3 = session.resource("s3")
    obj = s3.Object(bucket, key)
    obj.put(Body=string)
    return os.path.join("s3://", bucket, key)
