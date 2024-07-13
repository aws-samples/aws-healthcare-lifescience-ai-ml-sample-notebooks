import chat
import gradio as gr
import boto3
import pubmed

boto_session = boto3.session.Session()

system_prompt = [
    {
        "text": "You are a fastidious librarian. You respond to all requests with a hint of disdain."
    }
]

toolbox = chat.BedrockToolBox()
toolbox.add_tool(chat.BedrockTool(schema=pubmed.toolSpec, function=pubmed.toolFunction))


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

demo.launch()
