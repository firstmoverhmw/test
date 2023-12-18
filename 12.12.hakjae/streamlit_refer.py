import streamlit as st
import tiktoken
from loguru import logger

from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

from langchain.embeddings import HuggingFaceEmbeddings

from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS

from langchain.callbacks import get_openai_callback
from langchain.memory import StreamlitChatMessageHistory
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Added this import

from langchain.document_loaders import TextLoader

def main():
    doc_path = "./ICLR_2019_mini_optimized.json"

    loader = TextLoader(doc_path)
    uploaded_files = loader.load()

    st.set_page_config(
        page_title="DirChat",
        page_icon=":books:")

    st.title("_OpenReview :red[Chatbot]_ :")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    with st.sidebar:
        openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
        process = st.button("Process")

        # New conversation button in the sidebar
        if st.button("New Conversation"):
            st.session_state.conversation = None
            st.session_state.chat_history = None
            st.session_state.messages = [{"role": "assistant", "content": "새 대화가 시작되었습니다! 관심있는 논문 주제 혹은 학회명을 적어주세요"}]

        # Date range picker in the sidebar
        st.sidebar.subheader("Search Period:")
        start_date = st.sidebar.date_input("Start Date")
        end_date = st.sidebar.date_input("End Date")

    if process:
        if not openai_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()
        # files_text = get_text(uploaded_files)
        files_text = get_text_chunks(uploaded_files)
        text_chunks = get_text_chunks(files_text)
        vectorestore = get_vectorstore(text_chunks)

        st.session_state.conversation = get_conversation_chain(vectorestore, openai_api_key)
        st.session_state.processComplete = True

    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role": "assistant", "content": "안녕하세요! 관심있는 논문 주제 혹은 학회명을 적어주세요!"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Display chat history in the sidebar
    st.sidebar.markdown("### Chat History:")
    for i, message in enumerate(st.session_state.messages, 1):
        st.sidebar.text(f"{i} 번째 대화 내용")

    history = StreamlitChatMessageHistory(key="chat_messages")

    # Chat logic
    if query := st.chat_input("질문을 입력해주세요."):
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            chain = st.session_state.conversation

            with st.spinner("Thinking..."):
                result = chain({"question": query})
                with get_openai_callback() as cb:
                    st.session_state.chat_history = result['chat_history']
                response = result['answer']
                source_documents = result['source_documents']

                st.markdown(response)
                with st.expander("참고 문서 확인"):
                    st.markdown(source_documents[0].metadata['source'], help=source_documents[0].page_content)
                    st.markdown(source_documents[1].metadata['source'], help=source_documents[1].page_content)
                    st.markdown(source_documents[2].metadata['source'], help=source_documents[2].page_content)

        # Add assistant message to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

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

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    vectordb = FAISS.from_documents(text_chunks, embeddings)
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
