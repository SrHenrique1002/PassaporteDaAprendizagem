# 🔬 Ferramenta de Triagem Curricular por LLM

[![Licença](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Tecnologia Principal](https://img.shields.io/badge/Tecnologia-Python%20%7C%20Streamlit-blueviolet.svg)]()
[![Modelo LLM](https://img.shields.io/badge/LLM-Gemini%202.5%20Flash-0F7EF6.svg)]()

> 

### Fluxo RAG de Alto Nível


## 🛠️ Tecnologias Utilizadas

O projeto é construído em Python e utiliza as seguintes bibliotecas e serviços:

* **Linguagem:** `Python`
* **Frontend/Web App:** `Streamlit`
* **Orquestração de LLM:** `LangChain`
    * `langchain-google-genai`: Integração com modelos Google Gemini (LLM e Embeddings).
    * `FAISS`: Vector Store para indexação rápida dos documentos.
    * `PyPDFLoader`: Leitura de arquivos PDF.
    * `RecursiveCharacterTextSplitter`: Divisão eficiente de texto em chunks.
* **Variáveis de Ambiente:** `python-dotenv`

---

## ⚙️ Instalação

Siga os passos abaixo para configurar e rodar o projeto localmente.

### Pré-requisitos

1.  **Python 3.9+**
2.  **Chave de API do Gemini (Google AI Studio):** Você precisa de uma chave de API para o modelo Gemini.
3.  **Git** (para clonar o repositório)

### 1. Configurar o Ambiente

1.  Clone o repositório:
    ```bash
    git clone [https://github.com/SeuUsuario/NomeDoSeuProjeto.git](https://github.com/SeuUsuario/NomeDoSeuProjeto.git)
    cd NomeDoSeuProjeto
    ```
2.  Crie um ambiente virtual (recomendado) e ative-o:
    ```bash
    python -m venv venv
    source venv/bin/activate  # No Windows, use: venv\Scripts\activate
    ```
3.  Instale as dependências:
    ```bash
    pip install -r requirements.txt
    # Nota: Certifique-se de incluir todas as bibliotecas usadas (streamlit, langchain, etc.)
    ```

### 2. Configurar a Chave de API

1.  Crie um arquivo chamado `.env` na raiz do projeto.
2.  Insira sua chave de API do Gemini/Google, conforme o padrão esperado pelo `langchain_google_genai`:
    ```
    GEMINI_API_KEY="SUA_CHAVE_DE_API_DO_GEMINI_AQUI"
    ```

### 3. Executar a Aplicação

Inicie o aplicativo Streamlit:

```bash
streamlit run nome_do_arquivo_principal.py  # Ex: streamlit run main.py

O aplicativo será aberto automaticamente no seu navegador padrão.

## 🚀 Como Usar
O fluxo da aplicação é dividido em três passos automatizados:

Passo 1: Inicialização e Indexação (Automático)
Ao rodar a aplicação, o Streamlit verifica o diretório curriculos_base e:

Verifica: Se os índices FAISS (Base de Conhecimento) para os PDFs já existem.

Indexa: Caso não existam, a aplicação automaticamente processa os PDFs, divide o texto e cria a Base de Conhecimento (Vector Store FAISS) para cada ano.

Verificação: Confirme na Sidebar a mensagem Currículos indexados e prontos para consulta... para prosseguir.

Passo 2: Upload do Boletim e Análise
Na seção principal, faça o Upload do Boletim do Estudante (PDF).

Clique no botão Analisar Boletim e Gerar Questionário.

A aplicação executará duas cadeias de LLM em sequência:

Análise Estruturada: O LLM lerá o PDF e retornará uma saída JSON estruturada identificando o Ano Letivo (ex: 7º ano) e a Defasagem Específica em Matemática (ex: Frações).

RAG e Geração: O sistema buscará o currículo exato para o ano e tópico identificados e usará o LLM para gerar o Pré-Questionário de Diagnóstico.

Passo 3: Auditoria do Contexto
O Pré-Questionário gerado será exibido na tela. Utilize o Expander de Auditoria (Contexto Curricular Utilizado para Geração) para verificar o conteúdo curricular exato que foi recuperado pelo RAG para gerar as perguntas.

## 🤝 Contribuição
Contribuições são bem-vindas! Se você tiver sugestões ou quiser melhorar o projeto, por favor, siga o fluxo padrão de contribuição:

Faça um Fork do projeto.

Crie uma nova branch para sua feature (git checkout -b feature/minha-feature).

Faça commit das suas alterações (git commit -m 'feat: Adiciona melhoria X').

Faça Push para a branch (git push origin feature/minha-feature).

Abra um Pull Request.

