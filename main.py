"""
Main file for the Machine Translation App.
"""

import streamlit as st
from utils import predict
from utils import languages

st.set_page_config(
    page_title="Machine Translation App 🤖",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
#MainMenu {display: none;}
footer {display: none;}
.stDeployButton {display: none;}
</style>
""",
    unsafe_allow_html=True,
)

st.title("Machine Translation App 🤖")
st.subheader("Translate text to English using Seq2Seq Machine Translation.")

st.selectbox(
    "Select language code:",
    languages.keys(),
    key="language",
    placeholder="Deutsch (German)",
    index=1,
)

st.text_input(
    "Enter text to translate:", key="text_input", placeholder="Geh", value="Geh"
)

translated = predict(st.session_state.text_input, languages[st.session_state.language])

st.write(f"Translated text: {translated}")
