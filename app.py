import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax
import torch

# Carregar o modelo e o tokenizer
st.set_page_config(page_title="Verificar Sentimento", page_icon="🤖", layout="centered")
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# Definindo a função para classificar a intenção do texto
def classify_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = softmax(outputs.logits, dim=1)
    sentiment_index = torch.argmax(probs, dim=1).item()
    sentiment_label = 'Neutra' if sentiment_index == 2 else 'Positiva' if sentiment_index > 2 else 'Negativa'
    return sentiment_label

# Criando a interface Streamlit
st.title('Verificar Sentimento')
user_input = st.text_area("Escreva um comentário para analisar se é algo positivo, negativo ou neutro:", value="", height=150)
if st.button('Verificar texto'):
    sentiment = classify_sentiment(user_input)
    st.write(f"Essa frase é: **{sentiment}**")