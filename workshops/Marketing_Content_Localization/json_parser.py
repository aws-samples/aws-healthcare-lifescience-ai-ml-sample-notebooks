# Updating Langchain outputparser to enable strict=False loading 
# https://api.python.langchain.com/en/latest/_modules/langchain/output_parsers/json.html#parse_json_markdown

from langchain.output_parsers.json import _custom_parser
import re
import json

def parse_json_markdown(json_string: str, strict:bool=False) -> dict:
    match = re.search(r"```(json)?(.*)```", json_string, re.DOTALL)
    # If no match found, assume the entire string is a JSON string
    if match is None:
        json_str = json_string
    else:
        # If match found, use the content within the backticks
        json_str = match.group(2)
    # Strip whitespace and newlines from the start and end
    json_str = json_str.strip()
    # handle newlines and other special characters inside the returned value
    json_str = _custom_parser(json_str)

    # Parse the JSON string into a Python dictionary
    parsed = json.loads(json_str, strict=strict)

    return parsed
