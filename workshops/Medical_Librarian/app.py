import chat
import gradio as gr
import boto3
import pubmed

boto_session = boto3.session.Session()

system_prompt = [
    {
        "text": "You are an expert medical librarian trained to answer questions using scientific literature.",
        "text": "Please respond to all requests using a friendly tone.",
        "text": "Write all of your technical responses at a high school reading level."
    }
]

toolbox = chat.BedrockToolBox()
toolbox.add_tool(chat.BedrockTool(schema=pubmed.search_pubmed_spec, function=pubmed.search_pubmed))
toolbox.add_tool(chat.BedrockTool(schema=pubmed.get_full_text_spec, function=pubmed.get_full_text))


def respond(message, chat_history):
    messages = chat.gradio_to_bedrock(message, chat_history)
    response = chat.generate_text(
        messages=messages,
        toolbox=toolbox,
        system_prompts=system_prompt,
        boto_session=boto_session,
    )
    chat_history = chat.bedrock_to_gradio(response)

    return "", chat_history


with gr.Blocks() as demo:
    gr.Markdown("# Medical Librarian")
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot])
    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    examples = gr.Examples(
        examples=["Please search pubmed for recent articles about therapeutic enzyme engineering", "Please get the full text of pubmed article PMC8795449"],
        inputs=msg,
    )

demo.launch()
