#!/bin/bash

# Instala o Streamlit (se ainda não estiver instalado)
pip install streamlit

# Instala as dependências Python
pip install --no-cache-dir -r requirements.txt

# Inicia a aplicação Streamlit
streamlit run app.py
