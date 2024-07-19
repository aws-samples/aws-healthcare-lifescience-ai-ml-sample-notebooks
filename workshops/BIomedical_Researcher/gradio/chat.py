# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Shows how to use tools with the Converse API and the Cohere Command R model.
"""

import logging
import boto3
import json

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

boto_session = boto3.session.Session()

class BedrockToolError(Exception):
    """Raised when a requested tool isn't found."""

    pass

def generate_text(
    messages,
    boto_session=boto3.session.Session(),
    model_id="anthropic.claude-3-haiku-20240307-v1:0",
    toolbox=None,
    system_prompts=[],
):
    """Generates text using the supplied Amazon Bedrock model. If necessary,
    the function handles tool use requests and sends the result to the model.
    Args:
        messages (str): The input messages.
        bedrock_client: The Boto3 Bedrock runtime client.
        model_id (str): The Amazon Bedrock model ID.
        tool_config (dict): The tool configuration.
        system_prompts (list): The system prompts for the model.
    Returns:
        Nothing.
    """
    bedrock_client=boto_session.client(service_name="bedrock-runtime")

    logger.info("Generating text with model %s", model_id)
    logger.info(messages)


    if toolbox is not None:
        print(type(toolbox))
        print(toolbox)
        response = bedrock_client.converse(
            modelId=model_id,
            messages=messages,
            toolConfig=toolbox.get_tools(),
            system=system_prompts,
        )
    else:
        response = bedrock_client.converse(
            modelId=model_id,
            messages=messages,
            system=system_prompts,
        )

    output_message = response["output"]["message"]
    messages.append(output_message)
    stop_reason = response["stopReason"]

    if stop_reason == "tool_use":
        # Tool use requested. Call the tool and send the result to the model.
        tool_requests = response["output"]["message"]["content"]
        for tool_request in tool_requests:
            if "toolUse" in tool_request:
                tool = tool_request["toolUse"]
                tool_name = tool['name']
                tool_request = tool['toolUseId']
                tool_args = tool['input'] or {}
                logger.info(
                    f"Requesting tool {tool_name}. Request: {tool_request}"
                )
                logger.info(f"Tool arguments:\n{tool_args}")
                tool_response = toolbox.tools.get(tool_name)(**tool_args) or ""
                if tool_response:
                    tool_status = 'success'
                else:
                    tool_status = 'error'

                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                'toolResult': {
                                    'toolUseId': tool_request,
                                    'content': [
                                        {
                                            "text": json.dumps(tool_response)
                                        }
                                    ],
                                    'status': tool_status
                                }
                            }
                        ]
                    }
                )

                response = bedrock_client.converse(
                    modelId=model_id,
                    messages=messages,
                    toolConfig=toolbox.get_tools(),
                    system=system_prompts,
                )
                output_message = response["output"]["message"]
                messages.append(output_message)

    return messages

def gradio_to_bedrock(message, chat_history):
    logger.info("Converting to bedrock format")

    messages = []

    for turn in chat_history:
        for i, text in enumerate(turn):
            role = "user" if i % 2 == 0 else "assistant"
            if text:
                messages.append({"role": role, "content": [{"text": text}]})
            else:
                messages.append(
                    {"role": role, "content": [{"text": "Waiting for tool response"}]}
                )

    messages.append({"role": "user", "content": [{"text": message}]})
    return messages


def bedrock_to_gradio(messages):
    logger.info("Converting to gradio format")

    simple_messages = []
    for message in messages:
        content = {k: v for d in message.get("content") for k, v in d.items()}
        if "toolUse" in content or "toolResult" in content:
            continue
        else:
            simple_messages.append(
                {
                    "role": message.get("role"),
                    "text": message.get("content")[0].get("text", None),
                }
            )

    chat_history = []
    tmp = [None, None]
    for message in simple_messages:
        if message.get("role") == "user":
            tmp[0] = message.get("text")
        elif message.get("role") == "assistant":
            tmp[1] = message.get("text")
            chat_history.append(tmp)
            tmp = [None, None]

    return chat_history


class BedrockTool():
    """
    Class to handle Bedrock tool use requests
    """

    def __init__(self, schema, function) -> None:
        self.name = schema.get('name', 'default')
        self.schema = schema
        self.function = function

    def __call__(self, **kwargs):
        """
        Call the tool function
        """
        return self.function(**kwargs)

class BedrockToolBox():
    """
    Class to store Bedrock tools
    """

    def __init__(self) -> None:
        self.tools = {}

    def add_tool(self, tool: BedrockTool) -> bool:
        """
        Add a tool to the tool box
        """
        self.tools[tool.name] = tool
        return True
    
    def get_tools(self) -> dict:
        """
        Return a dictionary of tool shemas
        """
        toolConfig = {
            'tools': [],
            }
        
        for tool in self.tools.values():
            toolConfig['tools'].append({'toolSpec':tool.schema})

        return toolConfig