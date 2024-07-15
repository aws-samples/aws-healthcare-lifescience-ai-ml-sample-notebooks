import boto3
import botocore
import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Iterator, List
import xmltodict

logger = logging.getLogger(__name__)

##### Search PubMed Tool #####

search_pubmed_spec = {
            'name': 'search_pubmed',
            'description': 'Search for open access PubMed articles.',
            'inputSchema': {
                'json': {
                    'type': 'object',
                    'properties': {
                        'query': {
                            'type': 'string',
                            'description': 'Query terms for the PubMed search API'
                        },
                        'count': {
                            'type': 'integer',
                            'description': 'Number of PubMed records to return'
                        }
                    },
                    'required': ['query']
                }
            }
        }

def search_pubmed(query: str, count: int = 10, max_size: int = 8192, open_only: bool = True) -> str:
    """
    Run PubMed search and get the article meta information.
    """
    pm = PubMed()

    try:
        docs = []
        for result in pm.load(query[: pm.MAX_QUERY_LENGTH], count, open_only):
            if open_only and not result["PubMedId"]:
                continue                
            docs.append(
                {
                    "Published": result["Published"],
                    "Title": result["Title"],
                    "PubMedId": result["PubMedId"][0] if result["PubMedId"] else None,
                    "Summary": result["Summary"],
                }
            )
            if len(str(docs).encode()) > max_size:
                docs.pop()
                break

        logger.info(f"PubMed search results:")
        for doc in docs:
            logger.info(f"\n{doc['Title']}\n{doc['Published']}\n{doc['PubMedId']}\n{doc['Summary']}\n")
        # Join the results and limit the character count
        return docs if docs else "No good PubMed Result was found"
    except Exception as ex:
        return f"PubMed exception: {ex}"

##### Get Fulltext Tool #####

get_full_text_spec = {
            'name': 'get_full_text',
            'description': 'Get the full text of open pubmed central (PMC) articles from the registry of open data.',
            'inputSchema': {
                'json': {
                    'type': 'object',
                    'properties': {
                        'id': {
                            'type': 'string',
                            'description': 'Pubmed central ID, for example 8795449 or PMC8795449'
                        },
                    },
                    'required': ['id']
                }
            }
        }

def get_full_text(id: str) -> str:
    """
    Get the full text of open pubmed central (PMC) articles from the registry of open data.
    """
    
    id = id if id.startswith("PMC") else "PMC" + id

    S3_BUCKET_NAME = 'pmc-oa-opendata' 
    S3_KEY = f'oa_comm/txt/all/{id}.txt'

    s3_client = boto3.client('s3')

    try:
        s3_response_object = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=S3_KEY)
        fulltext = s3_response_object['Body'].read().decode()
    except botocore.exceptions.ClientError as error:
        if error.response['Error']['Code'] == 'NoSuchKey':
            return f"Fulltext for article {id} not found in the PubMed Central (PMC) article dataset on AWS Data Exchange."

    logger.info(f"Fulltext for article PMC{id}:")
    logger.info(fulltext)
    return fulltext

class PubMed:
    """
    Calls pubmed API to fetch biomedical literature.

    """

    base_url_esearch: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?"
    base_url_efetch: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?"
    max_retry: int = 5
    sleep_time: float = 0.2
    MAX_QUERY_LENGTH: int = 300

    def lazy_load(self, query: str, count: int, open_only: bool) -> Iterator[dict]:
        """
        Search PubMed for documents matching the query.
        Return an iterator of dictionaries containing the document metadata.
        """

        url = (
            self.base_url_esearch
            + "db=pubmed&term="
            + str({urllib.parse.quote(query)})
        )

        if open_only:
            url += "+AND+pmc+open+access[filter]"

        url += f"&retmode=json&retmax={count}&usehistory=y"
        
        result = urllib.request.urlopen(url)
        text = result.read().decode("utf-8")
        json_text = json.loads(text)

        webenv = json_text["esearchresult"]["webenv"]
        for uid in json_text["esearchresult"]["idlist"]:
            yield self.retrieve_article(uid, webenv)

    def load(self, query: str, count: int, open_only: bool) -> List[dict]:
        """
        Search PubMed for documents matching the query.
        Return a list of dictionaries containing the document metadata.
        """
        return list(self.lazy_load(query, count, open_only))

    def retrieve_article(self, uid: str, webenv: str) -> dict:
        url = self.base_url_efetch + "db=pubmed&retmode=xml&id=" + uid + "&webenv=" + webenv

        retry = 0
        while True:
            try:
                result = urllib.request.urlopen(url)
                break
            except urllib.error.HTTPError as e:
                if e.code == 429 and retry < self.max_retry:
                    # Too Many Requests errors
                    # wait for an exponentially increasing amount of time
                    logger.warning(
                        f"Too Many Requests, " f"waiting for {self.sleep_time:.2f} seconds..."
                    )
                    time.sleep(self.sleep_time)
                    self.sleep_time *= 2
                    retry += 1
                else:
                    raise e

        xml_text = result.read().decode("utf-8")
        text_dict = xmltodict.parse(xml_text)
        return self._parse_article(uid, text_dict)

    def _parse_article(self, uid: str, text_dict: dict) -> dict:
        try:
            ar = text_dict["PubmedArticleSet"]["PubmedArticle"]["MedlineCitation"]["Article"]
            pmid = [id.get('#text', None) for id in text_dict["PubmedArticleSet"]["PubmedArticle"]["PubmedData"]["ArticleIdList"]['ArticleId'] if id.get('@IdType') == "pmc"]
        except KeyError:
            ar = text_dict["PubmedArticleSet"]["PubmedBookArticle"]["BookDocument"]
        abstract_text = ar.get("Abstract", {}).get("AbstractText", [])
        summaries = [
            f"{txt['@Label']}: {txt['#text']}"
            for txt in abstract_text
            if "#text" in txt and "@Label" in txt
        ]
        summary = (
            "\n".join(summaries)
            if summaries
            else (
                abstract_text
                if isinstance(abstract_text, str)
                else (
                    "\n".join(str(value) for value in abstract_text.values())
                    if isinstance(abstract_text, dict)
                    else "No abstract available"
                )
            )
        )
        a_d = ar.get("ArticleDate", {})
        pub_date = "-".join([a_d.get("Year", ""), a_d.get("Month", ""), a_d.get("Day", "")])

        ## Reset throttling sleep timer after back off
        self.sleep_time = 0.2

        return {
            "uid": uid,
            "Title": ar.get("ArticleTitle", ""),
            "PubMedId": pmid,
            "Published": pub_date,
            "Copyright Information": ar.get("Abstract", {}).get("CopyrightInformation", ""),
            "Summary": summary,
        }
