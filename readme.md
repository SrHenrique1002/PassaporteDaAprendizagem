# 🔬 Ferramenta de Triagem Curricular por LLM

[![Licença](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Tecnologia Principal](https://img.shields.io/badge/Tecnologia-Python%20%7C%20Streamlit-blueviolet.svg)]()
[![Modelo LLM](https://img.shields.io/badge/LLM-Gemini%202.5%20Flash-0F7EF6.svg)]()

> Uma aplicação web (Streamlit) que utiliza a arquitetura **RAG (Retrieval-Augmented Generation)** para analisar currículos em PDF e gerar, automaticamente, pré-questionários de diagnóstico de 5 perguntas, focando na identificação de defasagens específicas em unidades curriculares.

## 🌟 Sobre o Projeto

Este projeto automatiza a criação de ferramentas de diagnóstico pedagógico. Ao invés de o professor ou analista ler manualmente extensos documentos curriculares para criar perguntas de avaliação, a aplicação faz o seguinte:

1.  **Indexa Currículos (PDF):** Utiliza o **LangChain** para carregar PDFs curriculares, dividi-los em pedaços (`chunks`) e indexá-los em um **VectorStore FAISS** com Embeddings do Google Gemini.
2.  **Busca Contextual (RAG):** O usuário insere a localização da defasagem (ex: "Matemática, II Unidade, 5º ano"). A aplicação busca o conteúdo programático exato (os `chunks` relevantes) dentro do VectorStore.
3.  **Geração com LLM:** Um `ChatPromptTemplate` instrui o modelo **Gemini 2.5 Flash** a agir como um "analista pedagógico" e gerar 5 perguntas de diagnóstico **apenas** com base no contexto curricular recuperado.

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
A interface do Streamlit divide o uso em duas etapas principais:

Passo 1: Indexação do Currículo (Sidebar)
Acesse a Sidebar (## Configuração Curricular (RAG)).

Clique em Browse files e faça o Upload do(s) Currículo(s) em PDF.

Clique no botão Indexar Currículo.

A aplicação irá processar os PDFs, dividir o texto e criar uma Base de Conhecimento (Vector Store FAISS). Aguarde a mensagem de sucesso Currículo indexado com sucesso!.

Passo 2: Geração do Questionário (Área Principal)
No campo de texto principal, insira a Localização da Defasagem de forma clara e específica.

Exemplo: Matemática, II Unidade, 5º ano

Exemplo: Língua Portuguesa, Gênero Textual Artigo de Opinião, 9º ano

Clique no botão Gerar Questionário.

O sistema executará a cadeia RAG, recuperará o conteúdo programático relevante e usará o LLM para gerar e exibir o Pré-Questionário de Diagnóstico diretamente na tela, formatado em Markdown.

## 🤝 Contribuição
Contribuições são bem-vindas! Se você tiver sugestões ou quiser melhorar o projeto, por favor, siga o fluxo padrão de contribuição:

Faça um Fork do projeto.

Crie uma nova branch para sua feature (git checkout -b feature/minha-feature).

Faça commit das suas alterações (git commit -m 'feat: Adiciona melhoria X').

Faça Push para a branch (git push origin feature/minha-feature).

Abra um Pull Request.

