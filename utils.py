from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from numpy import argmax
import pickle
import streamlit as st


languages = {
    "French": "fra",
    "Deutsch (German)": "deu",
    "español (Spanish)": "spa",
    "Italiano (Italian)": "ita",
    "Português": "por",
}


def get_tokenizer(language: str, english: bool = False):
    if english:
        with open(f"tokenizers/{language}/eng_tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)
        return tokenizer
    with open(f"tokenizers/{language}/{language}_tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return tokenizer


def encode_sequences(tokenizer, length, lines):
    seq = tokenizer.texts_to_sequences(lines)
    # pad sequences with 0 values
    seq = pad_sequences(seq, maxlen=length, padding="post")
    return seq


def tokenize(tokenizer, text: str):
    return encode_sequences(tokenizer, 12, [text])


def get_model(language: str):
    model = load_model(f"models/{language}.keras")
    return model


def get_word(n, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == n:
            return word
    return None


def get_text(prediction, language):
    eng_tokenizer = get_tokenizer(language, english=True)
    preds_text = []
    for i in prediction:
        temp = []
        for j in range(len(i)):
            t = get_word(i[j], eng_tokenizer)
            if j > 0:
                if (t == get_word(i[j - 1], eng_tokenizer)) or (t == None):
                    temp.append("")
                else:
                    temp.append(t)
            else:
                if t == None:
                    temp.append("")
                else:
                    temp.append(t)

        preds_text.append(" ".join(temp))
    return preds_text[0]


@st.cache_data()
def predict(text: str, language: str):
    tokenizer = get_tokenizer(language=language)
    model = get_model(language=language)

    seq = tokenize(tokenizer=tokenizer, text=text)

    preds = argmax(model.predict(seq.reshape((seq.shape[0], seq.shape[1]))), axis=-1)

    output = get_text(preds, language=language)

    return output
