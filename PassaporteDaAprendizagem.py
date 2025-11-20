import streamlit as sl
import os
import time

# Importações LangChain e Google Gemini
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

# Garante que o diretório 'uploaded' exista e carrega variáveis de ambiente
os.makedirs("uploaded", exist_ok=True) 
load_dotenv()

# ==============================================================================
# 1. FUNÇÕES DE CONFIGURAÇÃO DO LLM E RAG
# ==============================================================================

def load_prompt():
    """
    Carrega o template de prompt. A instrução força o LLM a primeiro identificar os tópicos 
    da unidade (usando o contexto RAG) e depois gerar as perguntas.
    """
    prompt = """Você é um analista pedagógico e criador de pré-questionários de diagnóstico.
Sua missão é criar um Pré-Questionário de 5 perguntas (e suas respostas) para identificar a defasagem exata de um estudante.

# FASE 1: ANÁLISE DE CONTEÚDO
1. O usuário forneceu a localização curricular (Disciplina, Unidade, Ano) na 'Pergunta'.
2. Use o 'Contexto Curricular Detalhado' para **priorizar e listar** os principais tópicos encontrados que correspondem a essa localização.

# FASE 2: GERAÇÃO DO PRÉ-QUESTIONÁRIO
1. Use **apenas** os tópicos listados na FASE 1 para criar 5 perguntas de múltipla escolha ou discursivas curtas.
2. O objetivo é testar os diferentes assuntos abordados naquela Unidade.
3. Formate o resultado usando Markdown e inclua um cabeçalho claro, citando a Unidade/Bimestre focado.

Contexto Curricular Detalhado: {context}
Pergunta (Localização da Defasagem): {question}

Se o contexto for insuficiente para identificar o conteúdo daquela unidade, responda "O Currículo indexado não detalha o conteúdo da unidade solicitada. Não é possível gerar o questionário."
"""
    prompt = ChatPromptTemplate.from_template(prompt)
    return prompt

def load_llm():
    """Carrega e retorna o modelo de linguagem (LLM) do Gemini."""
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    return llm

def format_docs(docs):
    """Formata os documentos recuperados em uma única string de contexto."""
    return "\n\n".join(doc.page_content for doc in docs)

# ==============================================================================
# 2. FUNÇÕES DE PROCESSAMENTO DE DADOS (RAG e Limpeza)
# ==============================================================================

def extract_data(pdf_docs):
    """Carrega, divide PDFs e cria o VectorStore FAISS."""
    text_chunks = []
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 512,
        chunk_overlap = 30,
        length_function = len,
        separators= ["\n\n", "\n", ".", " "]
    )
    
    for pdf in pdf_docs:
        save_uploadedfile(pdf) 
        loader = PyPDFLoader(os.path.join('uploaded', pdf.name))
        text_chunks.extend(loader.load_and_split(text_splitter=text_splitter))
        
    if not text_chunks: return None 
        
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004")
    except Exception:
        sl.error("Erro ao carregar o modelo de embeddings. Verifique a chave de API.", icon="❌")
        return None

    vectorstore = FAISS.from_documents(documents=text_chunks, embedding=embeddings)
    return vectorstore

def initialize_session_state():
    """Inicializa o estado de sessão da base de conhecimento."""
    if "knowledge_base" not in sl.session_state: sl.session_state["knowledge_base"] = None

def save_uploadedfile(uploadedfile):
    """Salva o arquivo PDF enviado na pasta 'uploaded'."""
    with open(os.path.join("uploaded", uploadedfile.name), "wb") as f: f.write(uploadedfile.getbuffer())

def remove_files():
    """Remove arquivos PDF do diretório 'uploaded'."""
    path = os.path.join(os.getcwd(), 'uploaded')
    for file_name in os.listdir(path):
        file = os.path.join(path, file_name)
        if os.path.isfile(file) and file.endswith(".pdf"): os.remove(file)

# ==============================================================================
# 3. INTERFACE STREAMLIT E LÓGICA PRINCIPAL
# ==============================================================================

if __name__ == '__main__':
    
    initialize_session_state()
    
    sl.title("🔬 Ferramenta de Triagem Curricular por LLM")
    
    # --- Sidebar (Upload e Processamento) ---
    with sl.sidebar:
        sl.markdown("## 📚 Configuração Curricular (RAG)")
        pdf_docs = sl.file_uploader(label="**1. Upload do(s) Currículo(s) em PDF:**", accept_multiple_files=True, type=["pdf"])
        submitted_pdf = sl.button("Indexar Currículo")
            
        if submitted_pdf:
            if pdf_docs:
                with sl.spinner("Processando Currículo (RAG Indexing)..."):
                    remove_files() 
                    sl.session_state.knowledge_base = extract_data(pdf_docs)
                if sl.session_state.knowledge_base:
                    sl.success("Currículo indexado com sucesso!", icon="✅")
                else:
                    sl.error("Erro na indexação. Verifique o PDF e a chave de API.", icon="❌")
            
            time.sleep(3) 

    # --- Aplicação Principal (Geração do Pré-Questionário) ---
    
    llm=load_llm()
    prompt=load_prompt()
    
    # Input principal: O usuário insere a localização da defasagem manualmente.
    query = sl.text_input(
        label='**2. Localização da Defasagem:**', 
        placeholder="Ex: Matemática, II Unidade, 5º ano (para buscar o conteúdo programático desse período)."
    )
    
    if sl.session_state.get("knowledge_base") is None:
        # Mostra a mensagem de aviso caso o currículo não tenha sido indexado
        sl.warning("Por favor, faça o Upload e Indexação do Currículo (Passo 1) para continuar.", icon="🚨")
    else:
        # Formulário de Geração
        sl.success("Currículo pronto para consulta. Insira a localização da defasagem abaixo.", icon="✅")
        
        if query:
            if sl.button("Gerar Questionário"):
                try:
                   
                    retriever = sl.session_state.knowledge_base.as_retriever(search_kwargs={"k": 2}) 
                    
                    # Cadeia RAG: Usa a QUERY (Localização) para buscar no Contexto (PDF)
                    rag_chain = (
                        {"context": retriever | format_docs, "question": RunnablePassthrough()}
                        | prompt
                        | llm
                        | StrOutputParser()
                    )
                    
                    with sl.spinner("Buscando conteúdo curricular e gerando questionário..."):
                        response = rag_chain.invoke(query)
                    
                    # Output
                    sl.subheader("3. Pré-Questionário de Diagnóstico Gerado:")
                    sl.markdown(response)
                    
                    sl.info(f"O questionário foi gerado com sucesso com base no conteúdo da unidade solicitada: **{query}**.", icon="💡")
                    
                except Exception as e:
                    print(f"Ocorreu um erro durante a execução do RAG: {e}")
                    sl.error("Ocorreu um erro ao processar sua solicitação. Verifique sua chave de API ou se a busca falhou (tente formatar melhor o PDF).", icon="❌")