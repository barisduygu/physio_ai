import os
import time
import pickle
import random
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

# import langchain
# langchain.debug = True

VECTORSTORE_DIR = "vectorstore"


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks, embeddings_selection):
    if embeddings_selection == "OpenAI":
        embeddings = OpenAIEmbeddings()
    else:
        embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")

    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_llm(llm_selection):
    if llm_selection == "OpenAI":
        llm = ChatOpenAI()
    else:
        llm = HuggingFaceHub(
            repo_id="google/flan-t5-xxl",
            model_kwargs={"temperature": 0.5, "max_length": 512},
        )

    return llm


def get_conversation_chain(vectorstore, llm):
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vectorstore.as_retriever(), memory=memory
    )
    return conversation_chain


def handle_userinput(user_question, llm_selection):
    cost = None  # Initialize cost here
    # if vectorstore_selection has a file name, load the vectorstore
    if st.session_state.vectorstore_selection != "Create New":
        vectorstore = load_vectorstore(st.session_state.vectorstore_selection)
        # Create conversation chain
        llm = get_llm(llm_selection)
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
    st.session_state.chat_history.extend(new_chat_history)

    # Scroll to the bottom of the page after each bot reply
    st.write(
        "<script>window.scrollTo(0,document.body.scrollHeight);</script>",
        unsafe_allow_html=True,
    )
    return cost  # Return cost here


def load_vectorstore(filename):
    with open(os.path.join(VECTORSTORE_DIR, filename), "rb") as file:
        vectorstore = pickle.load(file)
    return vectorstore


def save_vectorstore(vectorstore, filename):
    with open(os.path.join(VECTORSTORE_DIR, filename), "wb") as file:
        pickle.dump(vectorstore, file)

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    # Create the vectorstore directory if it doesn't exist
    os.makedirs(VECTORSTORE_DIR, exist_ok=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "vectorstore_selection" not in st.session_state:
        st.session_state.vectorstore_selection = "Create New"

    st.header("Chat with multiple PDFs :books:")

    with st.sidebar:
        st.subheader(":gear: Options")

        # Let the user choose the models
        llm_selection = st.selectbox(
            ":robot_face: Choose a Large Language Model",
            options=["OpenAI", "Falcon-40B-Instruct"],
        )
        embeddings_selection = st.selectbox(
            ":brain: Choose an Embeddings Model",
            options=["OpenAI", "HuggingFaceInstruct"],
        )

        # Let the user choose a vector store file, or create a new one
        vectorstore_files = ["Create New"] + os.listdir(VECTORSTORE_DIR)
        st.session_state.vectorstore_selection = st.selectbox(
            ":file_folder: Choose a Vector Store File", options=vectorstore_files
        )

        # Handle file upload
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'",
            type=["pdf", "txt"],
            accept_multiple_files=True,
        )
        if st.button("Process"):
            with st.spinner("Processing"):
                # Get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # Get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # Create or load vector store
                if (
                    st.session_state.vectorstore_selection == "Create New"
                    or not os.path.exists(
                        os.path.join(
                            VECTORSTORE_DIR, st.session_state.vectorstore_selection
                        )
                    )
                ):
                    vectorstore = get_vectorstore(text_chunks, embeddings_selection)
                    vectorstore_filename = f"{llm_selection}_{embeddings_selection}_{len(os.listdir(VECTORSTORE_DIR))}.pkl"
                    save_vectorstore(vectorstore, vectorstore_filename)
                    st.session_state.vectorstore_selection = vectorstore_filename  # update the current selection to the new file
                else:
                    vectorstore = load_vectorstore(
                        st.session_state.vectorstore_selection
                    )
                    vectorstore.update(text_chunks)

                # Create conversation chain
                llm = get_llm(llm_selection)
                st.session_state.conversation = get_conversation_chain(vectorstore, llm)

        if st.button("Clear Chat"):
            st.session_state.chat_history = []

    # user input box
    user_question = st.text_input(
        "Ask a question about your documents:", key="user_input"
    )

    if user_question:
        cost = handle_userinput(user_question, llm_selection)

    # Add a divider for the user input box
    st.markdown("---")

    # Display chat history
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(
                user_template.replace("{{MSG}}", message.content),
                unsafe_allow_html=True,
            )
        else:
            reply = bot_template.replace("{{MSG}}", message.content)
            if "cost" in locals():  # Check if cost is defined
                reply = reply.replace("{{COST}}", cost)

            st.write(reply, unsafe_allow_html=True)


if __name__ == "__main__":
    main()