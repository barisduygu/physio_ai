import os
import streamlit as st
from dotenv import load_dotenv
from PIL import Image
from pathlib import Path
import streamlit_authenticator as stauth

from utils import load_vectorstore, get_text_chunks, save_vectorstore, handle_userinput, get_conversation_chain
from htmlTemplates import css, bot_template, user_template

image = Image.open("assets/logo/logo_physio.png")
VECTORSTORE_DIR = "vectorstore"

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