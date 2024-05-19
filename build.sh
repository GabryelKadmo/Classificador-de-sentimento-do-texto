#!/bin/bash

# Define o caminho para o executável Python
PYTHON_EXEC=$(which python)

# Instala o Streamlit (se ainda não estiver instalado)
$PYTHON_EXEC -m pip install streamlit

# Instala as dependências Python
$PYTHON_EXEC -m pip install --no-cache-dir -r requirements.txt

# Inicia a aplicação Streamlit
$PYTHON_EXEC -m streamlit run app.py
