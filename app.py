import os
import pickle
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
from langchain.callbacks import get_openai_callback
import openai
from PIL import Image
from streamlit_extras.add_vertical_space import add_vertical_space
from pathlib import Path

image = Image.open("./assets/logo/logo_physio.png")
VECTORSTORE_DIR = "vectorstore"


def load_vectorstore(filename):
    with open(os.path.join(VECTORSTORE_DIR, filename), "rb") as file:
        vectorstore = pickle.load(file)
    return vectorstore


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def save_vectorstore(vectorstore, filename):
    with open(os.path.join(VECTORSTORE_DIR, filename), "wb") as file:
        pickle.dump(vectorstore, file)


def handle_userinput(user_question):
    cost = None
    if st.session_state.vectorstore_selection != "Leitlinie auswählen":
        vectorstore = load_vectorstore(st.session_state.vectorstore_selection)
        llm = ChatOpenAI()
        st.session_state.conversation = get_conversation_chain(vectorstore, llm)
    elif (
        "conversation" not in st.session_state or st.session_state.conversation is None
    ):
        st.warning(
            "Please upload a document and click 'Process' before asking questions."
        )
        return cost
    with get_openai_callback() as cb:
        response = st.session_state.conversation({"question": user_question})
        cost = str(cb.total_cost) if cb is not None else "0"
        cost = "{:.5f}".format(float(cost))
    new_chat_history = response["chat_history"]

    # Check if the response is empty or not relevant
    if not new_chat_history or not any(message.content for message in new_chat_history):
        new_chat_history = [
            {
                "content": "Tut mir leid, bitte stellen Sie eine Frage bzgl. der Leitlinien"
            }
        ]

    st.session_state.chat_history.extend(new_chat_history)

    st.write(
        "<script>window.scrollTo(0,document.body.scrollHeight);</script>",
        unsafe_allow_html=True,
    )
    return cost


def get_conversation_chain(vectorstore, llm):
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vectorstore.as_retriever(), memory=memory
    )
    return conversation_chain


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")

    os.makedirs(VECTORSTORE_DIR, exist_ok=True)
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "vectorstore_selection" not in st.session_state:
        st.session_state.vectorstore_selection = "Leitlinie auswählen"

    with st.sidebar:
        st.image(image)

        st.markdown(
            "<h1 style='text-align: center;; color: grey; font-size: 15px;'>PhysioAI hilft Ihnen dabei gezielt Informationen aus den Leitlinien zu finden.</h1>",
            unsafe_allow_html=True,
        )

        add_vertical_space(2)
        vectorstore_files = ["Leitlinie auswählen"] + os.listdir(VECTORSTORE_DIR)
        st.session_state.vectorstore_selection = st.selectbox(
            ":file_folder: Wählen Sie die Leitlinie aus", options=vectorstore_files
        )

    user_question = st.text_input("Stellen Sie hier Ihre Fragen:", key="user_input")

    if user_question:
        cost = handle_userinput(user_question)

    st.markdown("---")

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(
                user_template.replace("{{MSG}}", message.content),
                unsafe_allow_html=True,
            )
        else:
            reply = bot_template.replace("{{MSG}}", message.content)
            if "cost" in locals():
                reply = reply.replace("{{COST}}", cost)

            st.write(reply, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
