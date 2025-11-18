import gradio as gr
from landgraph_agent import chatbot_response

def say_hi(message, __):
    print(message)
    yield "HI, I'm a chatbot, how can I assist you today?"


demo = gr.ChatInterface(
    chatbot_response,
    chatbot=gr.Chatbot(height=600, type='messages'),
    textbox=gr.Textbox(placeholder="Ask chatbot...", container=False, scale=7),
    title="Chatbot",
    description="A simple chatbot.",
    theme="soft",
    examples=[["What is the capital of France?"], ["Explain quantum computing simply."]],
    cache_examples=False,
    type='messages'
)


if __name__ == "__main__":
    demo.launch()