from llama_index.core import download_loader, Settings
from llama_index.core.agent import FunctionCallingAgentWorker, AgentRunner
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex
from llama_index.core.objects import ObjectIndex
import openai 
import streamlit as st


import os
import time
import sys
import io
from utils import get_tools

import nest_asyncio
nest_asyncio.apply()

FOLDER_PATH = 'data'

st.subheader(':robot_face: Multi-Document Agentic RAG for Quantum Computing')
st.caption("Created by [Jay Shah]('https://www.linkedin.com/in/jay-shah-qml/') with :heart:")






with st.sidebar:
    files = os.listdir(FOLDER_PATH)
    expander = st.expander("See Documents")
    for file in files:
        expander.write(file)
    OPENAI_API_KEY = st.text_input("OPENAPI KEY", key="chatbot_api_key", type="password")
    if OPENAI_API_KEY:
        openai.api_key = OPENAI_API_KEY
    url = "https://www.streamlit.io"
    verbose_toggle = st.toggle("Verbose")
    reset = st.button('Reset Chat!')
    st.write("[Get your API key](https://platform.openai.com/account/api-keys)")
    st.write("[Source Code](%s)" % url)
    



if OPENAI_API_KEY:
    if "tools_loaded" not in st.session_state:
        try:
            # load openai model
            model = OpenAI()

            # get path to jupyter notebooks
            doc_paths = {
            "bernstein-vazirani": r"data/bernstein-vazirani algorithm.ipynb",
            "deutsch-jozsa":r"data/deutsch-jozsa algorithm.ipynb",
            "grover":r"data/grover's algorithm.ipynb",
            "quantum-phase-estimation":r"data/quantum-phase-estimation algorithm.ipynb",
            "shor":r"data/shor's algorithm.ipynb",
            "simon":r"data/simon's algorithm.ipynb"
            } 

            # Download and initialize the IPYNBReader loader
            IPYNBReader = download_loader("IPYNBReader")
            loader = IPYNBReader(concatenate=True)

            # get the tools
            tools = get_tools(doc_paths=doc_paths,loader=loader,llm=model)


            obj_index = ObjectIndex.from_objects(
                tools,
                index_cls=VectorStoreIndex,
            ) 
            obj_retriever = obj_index.as_retriever(similarity_top_k=3)


            agent_worker = FunctionCallingAgentWorker.from_tools(
                tool_retriever=obj_retriever, 
                llm=model, 
                verbose=True
            )
            agent = AgentRunner(agent_worker)
            st.session_state["tools_loaded"] = True
            st.session_state["agent"] = agent
        except Exception as e:
            st.error(e)





if "messages" not in st.session_state or reset==True:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    if not OPENAI_API_KEY:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    try:
        with st.spinner('Wait for output...'):
            # Redirect stdout
            original_stdout = sys.stdout
            sys.stdout = io.StringIO()

            # Run your function here
            response = st.session_state.agent.query(prompt)


            # Get the captured output and restore stdout
            output = sys.stdout.getvalue()
            sys.stdout = original_stdout

            verbose = ''
            for output_string in output.split('==='):
                verbose+=output_string
                verbose+='\n'

            msg = f'{verbose}' if verbose_toggle else f'{response.response[10:]}'
        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.chat_message("assistant").write(msg)
    except Exception as e:
        st.error(e)


