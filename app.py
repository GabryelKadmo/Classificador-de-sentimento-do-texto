import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax
import torch

# Carregar o modelo pré-treinado e o tokenizer
st.set_page_config(page_title="Classificador de Sentimentos", page_icon=":speech_balloon:", layout="centered")
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# Definindo a função para classificar o sentimento
def classify_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = softmax(outputs.logits, dim=1)
    sentiment_index = torch.argmax(probs, dim=1).item()
    sentiment_label = 'Neutro' if sentiment_index == 2 else 'Positivo' if sentiment_index > 2 else 'Negativo'
    return sentiment_label

# Criando a interface Streamlit
st.title('Classificador de Sentimentos')
user_input = st.text_area("Digite uma frase para analisar o sentimento:", value="", height=150)
if st.button('Classificar Sentimento'):
    sentiment = classify_sentiment(user_input)
    st.write(f"O sentimento desta frase é: **{sentiment}**")