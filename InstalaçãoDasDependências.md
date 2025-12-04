# 📦 Instalação das Dependências — Guia Rápido

## 1️⃣ Criar ambiente virtual

```bash
python -m venv venv
```

## 2️⃣ Ativar ambiente virtual

```bash
venv\Scripts\activate
```

## 3️⃣ Instalar dependências individualmente

```bash
pip install streamlit
pip install langchain
pip install langchain-core
pip install langchain-community
pip install langchain-google-genai
pip install pypdf
pip install faiss-cpu
pip install python-dotenv
```

## 4️⃣ Instalar tudo via `requirements.txt`

Crie o arquivo:

```
streamlit
langchain
langchain-core
langchain-community
langchain-google-genai
pypdf
faiss-cpu
python-dotenv
```

E instale:

```bash
pip install -r requirements.txt
```

## 5️⃣ Verificar instalação

```bash
pip list
```

## 6️⃣ Testar FAISS

```bash
python -c "import faiss; print('FAISS OK')"
```

---
