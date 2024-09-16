import boto3
from dotenv import load_dotenv
import gradio as gr
import json
import logging
import os
import re
from tqdm import tqdm
from timeit import default_timer as timer

from json.decoder import JSONDecodeError

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

session = boto3.session.Session()
agents_runtime_client = session.client("bedrock-agent-runtime")

load_dotenv()


def format_search_results(input_list):
    response_string = f"{len(input_list)} documents found"
    for i, item in enumerate(input_list, start=1):
        response_string += f"<h2>{i}. {item.get('Title')}</h2>"
        response_string += f'<p><span class="search-result-header">Confidence:</span> {item.get("Confidence")}</p>'
        response_string += f'<p><span class="search-result-header">Document:</span> <a href="{item.get("Uri")}" target="_blank">{item.get("Uri")}</a></p>'
        response_string += f'<p><span class="search-result-header">Category:</span> {item.get("Category")}</p>'
        response_string += f'<p><span class="search-result-header">Manufacturer Name:</span> {item.get("ManufacturerName")}</p>'
        response_string += f'<p><span class="search-result-header">Brand Name:</span> {item.get("BrandName")}</p>'
        response_string += f'<p><span class="search-result-header">Generic Name:</span> {item.get("GenericName")}</p>'
        response_string += f'<p><span class="search-result-header">Application Number:</span> {item.get("ApplicationNumber")}</p>'
        response_string += f'<p><span class="search-result-header">Submission:</span> {item.get("Submission")}</p>'
        response_string += f'<p><span class="search-result-header">Excerpt:</span></p><blockquote>{item.get("Content")}</blockquote>'
        response_string += "Submit feedback: 👍 👎"
        response_string += "<hr />"
        response_string += "\n\n"

    html_string = (
        '<!DOCTYPE html><html><head><meta http-equiv="content-type" content="text/html; charset=UTF-8" /></head><body>'
        + response_string
        + "</body></html>"
    )
    return html_string


def format_generate_results(search_results, generate_results):
    search_uri_list = [result.get("Uri") for result in search_results]

    if search_uri_list == []:
        return '<!DOCTYPE html><html><head><meta http-equiv="content-type" content="text/html; charset=UTF-8" /></head><body></body></html>'
    response_string = ""
    try:
        for part in json.loads(generate_results):
            response_string += part.get("text")
            response_string += "<sup>"

            for source in part.get("sources"):
                try:
                    response_string += f'<a href="{source}" target="_blank">{search_uri_list.index(source) + 1}</a> '
                except Exception as e:
                    logger.info(e)
                    continue
            response_string += "</sup>"

        html_string = (
            '<!DOCTYPE html><html><head><meta http-equiv="content-type" content="text/html; charset=UTF-8" /></head><body>'
            + '<div class="border-gradient border-gradient generate-results">'
            + '<div class="logo-table"><div><img src="/file/www/img/brain-light.png" /></div><div class="bedrock-title">Amazon Bedrock Summary</div></div><br />'
            + '<p class="p-generate">'
            + response_string
            + "</p>"
            + "</div>"
            + "</body></html>"
        )
    except JSONDecodeError as e:
        logger.error(f"Error decoding JSON: {e}")
        html_string = '<!DOCTYPE html><html><head><meta http-equiv="content-type" content="text/html; charset=UTF-8" /></head><body></body></html>'
    return html_string


def get_filter_data(search_results):
    filter_data = {
        "Category": set(),
        "BrandName": set(),
        "GenericName": set(),
        "ManufacturerName": set(),
    }

    for result in search_results:
        for filter in filter_data.keys():
            filter_data[filter].add(result.get(filter, "None"))

    return filter_data


def get_filter(label="Category", choices=[]):
    checkgroup = gr.CheckboxGroup(
        label=label,
        elem_classes="filter-group",
        choices=choices,
        interactive=True,
        show_label=True,
        value=None,
    )
    return checkgroup


def submit_query(query):
    logger.info(os.getcwd())

    start_time = timer()
    logger.info(f"Starting execution")

    logger.info(f'Processing query "{query}"')

    invoke_response = agents_runtime_client.invoke_flow(
        flowAliasIdentifier=os.environ["FLOW_ALIAS_ID"],
        flowIdentifier=os.environ["FLOW_ID"],
        inputs=[
            {
                "content": {
                    "document": query,
                },
                "nodeName": "FlowInputNode",
                "nodeOutputName": "document",
            },
        ],
    )
    output = {}
    for event in invoke_response["responseStream"]:
        if (
            event.get("flowOutputEvent", {"nodeName": None}).get("nodeName")
            == "FlowOutputNode"
        ):
            logger.info(
                f'Generated response:\n{event["flowOutputEvent"]["content"]["document"]}'
            )
            output["generate"] = event["flowOutputEvent"]["content"]["document"]
        elif (
            event.get("flowOutputEvent", {"nodeName": None}).get("nodeName")
            == "SearchOutputNode"
        ):
            logger.info(
                f'{len(event["flowOutputEvent"]["content"]["document"])} documents returned from Kendra index'
            )

            output["search"] = event["flowOutputEvent"]["content"]["document"]

    filter_data = get_filter_data(output.get("search"))

    logger.info(f"Total execution time is {round(timer() - start_time, 3)}s")
    logger.info("Execution complete.")

    return [
        gr.HTML(
            visible=True,
            value=format_search_results(output.get("search")),
            show_label=False,
        ),
        gr.HTML(
            visible=True,
            value=format_generate_results(output.get("search"), output.get("generate")),
            show_label=False,
        ),
        get_filter("Category", filter_data["Category"]),
        get_filter("Brand Name", filter_data["BrandName"]),
        get_filter("Generic Name", filter_data["GenericName"]),
        get_filter("Manufacturer Name", filter_data["ManufacturerName"]),
        gr.Column(visible=True),
    ]


def get_root_query(query):
    return re.match(r"(.*?)(AND|$)", query).group().strip()


def filter_query(
    input_query,
    category_filter,
    brand_name_filter,
    generic_name_filter,
    manufacturer_name_filter,
):

    output_query = get_root_query(input_query)

    filter = {
        "Category": category_filter,
        "BrandName": brand_name_filter,
        "GenericName": generic_name_filter,
        "ManufacturerName": manufacturer_name_filter,
    }

    for k, v in filter.items():
        for filter_value in v:
            output_query += f" AND {k}:{filter_value}"
    return output_query
