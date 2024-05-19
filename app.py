import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax
import torch
from transformers import AutoConfig

# Carregar o modelo pré-treinado e o tokenizer
st.set_page_config(page_title="Verificar Texto", page_icon="🧠", layout="centered")
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
    sentiment_label = 'Neutra' if sentiment_index == 2 else 'Positiva' if sentiment_index > 2 else 'Negativa'
    return sentiment_label

# Criando a interface Streamlit
st.title('Verificar Sentimento')
user_input = st.text_area("Digite uma frase para analisar se é algo positivo, negativo ou neutro:", value="", height=150)
if st.button('Verificar texto'):
    sentiment = classify_sentiment(user_input)
    st.write(f"Essa frase é: **{sentiment}**")