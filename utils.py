import os
import pickle
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.callbacks import get_openai_callback

VECTORSTORE_DIR = "vectorstore"

def load_vectorstore(filename):
    with open(os.path.join(VECTORSTORE_DIR, filename), "rb") as file:
        vectorstore = pickle.load(file)
    return vectorstore

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def save_vectorstore(vectorstore, filename):
    with open(os.path.join(VECTORSTORE_DIR, filename), "wb") as file:
        pickle.dump(vectorstore, file)

def handle_userinput(user_question):
    cost = None
    if 'vectorstore_selection' not in st.session_state:
        st.session_state.vectorstore_selection = "Leitlinie auswählen"
        
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