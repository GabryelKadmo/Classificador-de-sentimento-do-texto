import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax
import torch
from transformers import AutoConfig

# Load model and tokenizer outside of the Streamlit app's main execution loop
config = AutoConfig.from_pretrained("bert-base-cased")
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# Function to classify sentiment
def classify_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = softmax(outputs.logits, dim=1)
    sentiment_index = torch.argmax(probs, dim=1).item()
    sentiment_label = 'Neutro' if sentiment_index == 2 else 'Positivo' if sentiment_index > 2 else 'Negativo'
    return sentiment_label

# Streamlit interface
st.set_page_config(page_title="Classificador de Sentimentos", page_icon=":speech_balloon:", layout="centered")
st.title('Classificador de Sentimentos')

# Main input form
user_input = st.text_area("Digite uma frase para analisar o sentimento:", value="", height=150)

# Store the button instance in a variable
classify_button = st.button('Classificar Sentimento')

# Perform sentiment analysis when button is clicked
if classify_button and user_input.strip():
    sentiment = classify_sentiment(user_input)
    st.write(f"O sentimento desta frase é: **{sentiment}**")
elif classify_button and not user_input.strip():
    st.warning("Por favor, digite uma frase para análise de sentimento.")
