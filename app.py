from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

import chainlit as cl
from chainlit.input_widget import Select, Slider, TextInput
import base64


from langgraph.graph import StateGraph
from langgraph.graph.message import MessagesState
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage


def call_model(state: MessagesState):
    messages = [SystemMessage(cl.user_session.get('system_message'))] + state['messages']
    llm = cl.user_session.get('llm')
    response = llm.invoke(messages)
    return {'messages': [response]}

builder = StateGraph(MessagesState)
builder.add_node('call_model', call_model)

builder.set_entry_point('call_model')
builder.set_finish_point('call_model')

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

# from IPython.display import Image, display
#
# try:
#     with open('graph.png', 'wb') as file:
#         file.write(Image(graph.get_graph().draw_mermaid_png()).data)
# except Exception:
#     # This requires some extra dependencies and is optional
#     pass

# Function to encode the image
def encode_image(image_path):
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

@cl.password_auth_callback
def auth_callback(username: str, password: str):
    if (username, password) == ('admin', 'admin'):
        return cl.User(
            identifier='admin', metadata={'role': 'admin', 'provider': 'credentials'}
        )
    else:
        return None

@cl.on_message
async def on_message(msg: cl.Message):

    config = {'callbacks': [cl.LangchainCallbackHandler()], 'configurable': {'thread_id': msg.thread_id}}
    final_answer = cl.Message(content='')

    content = msg.content
    if len(msg.elements) >0 :
        content = [ {'type': 'text', 'text': msg.content} ]
        for element in msg.elements:
            if element.type == 'image':
                base64_image = encode_image(element.path)
                content.append({'type': 'image_url', 'image_url': {'url': f'data:{element.mime};base64,{base64_image}'}})

    async for chunk, metadata in graph.astream(
            {'messages': [HumanMessage(content=content)]},
            stream_mode='messages',
            config=config
    ):
        await final_answer.stream_token(chunk.content)

    await final_answer.send()

async def setup_settings(restored_settings=None):
    models = ['gpt-4-turbo', 'chatgpt-4o-latest', 'gpt-4o', 'gpt-4o-mini', 'gpt-4', 'gpt-3.5-turbo']
    settings = await cl.ChatSettings(
        [
            Select(
                id='Model',
                label='OpenAI - Model',
                values=models,
                initial_index=0 if restored_settings is None else models.index(restored_settings.get('Model')),
            ),
            TextInput(
                id='System Message',
                label='OpenAI - System Message',
                initial = (
                    'You are a helpful assistant.' if restored_settings is None
                    else restored_settings.get('System Message')
                ),
                multiline=True,
            ),
            Slider(
                id='Temperature',
                label='OpenAI - Temperature',
                initial=0.7 if restored_settings is None else restored_settings.get('Temperature'),
                min=0,
                max=1,
                step=0.1,
            ),
        ]
    ).send()
    return settings

@cl.on_chat_start
async def start():
    settings = await setup_settings()
    await setup_agent(settings)

@cl.on_settings_update
async def setup_agent(settings):
    llm = ChatOpenAI(
        streaming=True,
        model_name=settings['Model'],
        temperature=settings['Temperature']
    )
    cl.user_session.set('llm', llm)
    cl.user_session.set('system_message', settings['System Message'])

@cl.on_chat_resume
async def on_chat_resume(thread):
    settings = cl.user_session.get('chat_settings')
    await setup_settings(settings)
    await setup_agent(settings)
