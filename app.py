import asyncio
from langchain_ollama.llms import OllamaLLM
from langgraph.checkpoint.memory import MemorySaver
import chainlit as cl
from chainlit.input_widget import Select, Slider, TextInput
import base64
from langgraph.graph import StateGraph
from langgraph.graph.message import MessagesState
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage

def call_model(state: MessagesState):
    messages = [HumanMessage(content=msg.content) if isinstance(msg, HumanMessage) else msg for msg in state['messages']]
    llm = cl.user_session.get('llm')
    response = llm.invoke(messages)
    return {'messages': [response]}

builder = StateGraph(MessagesState)
builder.add_node('call_model', call_model)
builder.set_entry_point('call_model')
builder.set_finish_point('call_model')
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

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
    if len(msg.elements) > 0:
        content = [{'type': 'text', 'text': msg.content}]
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
    models = ['deepseek-r1:14b', 'other-ollama-models']
    initial_model = 'deepseek-r1:14b' if restored_settings is None else restored_settings.get('Model')
    if initial_model not in models:
        initial_model = models[0]
    settings = await cl.ChatSettings(
        [
            Select(
                id='Model',
                label='Ollama - Model',
                values=models,
                initial_index=models.index(initial_model),
            ),
            TextInput(
                id='System Message',
                label='Ollama - System Message',
                initial=(
                    'You are a helpful assistant.' if restored_settings is None
                    else restored_settings.get('System Message')
                ),
                multiline=True,
            ),
            Slider(
                id='Temperature',
                label='Ollama - Temperature',
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
    llm = OllamaLLM(model=settings['Model'])
    cl.user_session.set('llm', llm)
    cl.user_session.set('system_message', settings['System Message'])

@cl.on_chat_resume
async def on_chat_resume(thread):
    settings = cl.user_session.get('chat_settings')
    await setup_settings(settings)
    await setup_agent(settings)
