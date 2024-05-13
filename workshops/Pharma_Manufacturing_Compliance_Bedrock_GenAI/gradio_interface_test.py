import gradio as gr

def greet(name):
    return "Hello " + name + "!"

def greet2(name2):
    return "Hello " + name2 + "!"

demo = gr.Interface(fn=greet, inputs="text", outputs="text")
demo = gr.Interface(fn=greet2, inputs="text", outputs="text")

demo.launch()   