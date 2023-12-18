__import__("pysqlite3")
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import re
import textwrap

import chromadb
from chromadb.config import Settings
import streamlit as st
import tiktoken

from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

from langchain.embeddings import HuggingFaceEmbeddings

from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Chroma

from langchain.callbacks import get_openai_callback
from langchain.memory import StreamlitChatMessageHistory
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Added this import

from langchain.document_loaders import TextLoader
from utils.parser import QueryParser

def main():
    parser = QueryParser()

    st.set_page_config(
        page_title="DirChat",
        page_icon=":books:")

    st.title("_OpenReview :red[Chatbot]_ :")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    with st.sidebar:
        openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
        process = st.button("Process")

        # Year selection buttons in the sidebar
        selected_years = st.multiselect("Select Year", ["2021", "2022", "2023"])
        
        # New conversation button in the sidebar
        if st.button("New Conversation"):
            st.session_state.conversation = None
            st.session_state.chat_history = None
            st.session_state.messages = [{"role": "assistant", "content": "새 대화가 시작되었습니다! 관심있는 논문 주제 혹은 학회명을 적어주세요"}]

    if process:
        if not openai_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()
        # files_text = get_text(uploaded_files)
        vectorestore = get_vectorstore()

        st.session_state.conversation = get_conversation_chain(vectorestore, openai_api_key)
        st.session_state.processComplete = True

    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role": "assistant", "content": "안녕하세요! 관심있는 논문 주제 혹은 학회명을 적어주세요!"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat logic
    if query := st.chat_input("질문을 입력해주세요."):
        
        if selected_years:
            query += " in " + " ".join(selected_years)
        st.session_state.messages.append({"role": "user", "content": query})
                                       

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            chain = st.session_state.conversation
            parsed_query = parser.parse_query(query)
            with st.spinner(f"Thinking...{parsed_query}"):
                result = chain({"question": query})
                
                with get_openai_callback() as cb:
                    st.session_state.chat_history = result['chat_history']
                response = result['answer']
                source_documents = result['source_documents']

                st.markdown(response)
                with st.expander("참고 문서 확인"):
                    for i in range(3):
                        render_reference(source_documents[i])

        # Add assistant message to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

def escape_markdown(text):
    return re.sub(r"[\\`*_{}[\]()#+.!-]", r"\\\g<0>", text)

def render_reference(document):
    authors = [author[1:-1] for author in document.metadata["authors"].split(", ")]
    if len(authors) > 3:
        authors = authors[:3]
        authors[-1] += " et al."
    metadata = textwrap.dedent(
        f"""\
        **{escape_markdown(document.metadata["title"])}**  
        {", ".join(authors)}

        :gray[*{escape_markdown(document.metadata["venue"])}*]
        """
    )
    content = document.page_content.split("\nAbstract:\n", maxsplit=1)[1]
    return st.markdown(metadata, help=content)

def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        length_function=tiktoken_len
    )
    chunks = text_splitter.split_documents(text)
    return chunks

def get_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/gtr-t5-large",
        model_kwargs={"device": "cpu"},
    )
    client = chromadb.HttpClient(
        host=st.secrets["CHROMA_HOST"],
        port=st.secrets["CHROMA_PORT"],
        settings=Settings(
            anonymized_telemetry=False,
            chroma_client_auth_provider="chromadb.auth.basic.BasicAuthClientProvider",
            chroma_client_auth_credentials=st.secrets["CHROMA_CREDENTIALS"],
        ),
    )
    vectordb = Chroma(
        client=client, collection_name="openreview_gtr", embedding_function=embeddings
    )
    return vectordb

def get_conversation_chain(vectorestore, openai_api_key):
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name='gpt-3.5-turbo', temperature=0)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=vectorestore.as_retriever(search_type='mmr', verbose=True),
        memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
        get_chat_history=lambda h: h,
        return_source_documents=True,
        verbose=True
    )
    return conversation_chain

if __name__ == '__main__':
    main()

