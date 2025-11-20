import streamlit as sl
import os
import time

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
# NOVO: Importa Embeddings do Google
from langchain_google_genai import GoogleGenerativeAIEmbeddings 
# NOVO: Importa o Chat do Google
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

# Garante que o diretório 'uploaded' exista
os.makedirs("uploaded", exist_ok=True) 
load_dotenv()

def load_prompt():
    """Carrega e retorna o template de prompt para a cadeia RAG."""
    prompt = """Você é um assistente que deve responder à Pergunta baseada no Contexto informado.
O contexto e a pergunta do utilizador são apresentados a seguir.
Contexto: {context}
Pergunta: {question}
Se a resposta não estiver no pdf, responda "Não consigo responder a essa pergunta com minha base de informações"
"""
    prompt = ChatPromptTemplate.from_template(prompt)
    return prompt

def load_llm():
    """Carrega e retorna o modelo de linguagem (LLM) do Gemini."""
    # NOVO: Usa o modelo Gemini Flash
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    return llm

def format_docs(docs):
    """Formata os documentos recuperados em uma única string de contexto."""
    return "\n\n".join(doc.page_content for doc in docs)

def extract_data():
    """
    Carrega, divide PDFs e cria o VectorStore FAISS usando embeddings do Google.
    """
    text_chunks = []
    
    # 1. Filtra e lista arquivos PDF no diretório 'uploaded'
    files = filter(lambda f: f.lower().endswith(".pdf"), os.listdir("uploaded"))
    file_list = list(files)
    
    # Configuração do Text Splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 512,
        chunk_overlap = 30,
        length_function = len,
        separators= ["\n\n", "\n", ".", " "]
    )
    
    # 2. Carrega e divide os documentos
    for file in file_list:
        loader = PyPDFLoader(os.path.join('uploaded', file))
        text_chunks.extend(loader.load_and_split(text_splitter=text_splitter))
        
    if not text_chunks:
        sl.warning("Nenhum PDF encontrado ou processado.", icon="⚠️")
        return None 
        
    # 3. Cria o VectorStore com Embeddings do Google
    try:
        # NOVO: Inicializa o embedding do Google
        embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004")
    except Exception as e:
        sl.error(f"Erro ao carregar o modelo de embeddings do Google: {e}. Verifique a variável GEMINI_API_KEY.", icon="❌")
        return None

    # Cria o VectorStore FAISS
    vectorstore = FAISS.from_documents(documents=text_chunks, embedding=embeddings)
    return vectorstore

def initialize_session_state():
    """Inicializa a base de conhecimento no estado da sessão do Streamlit."""
    if "knowledge_base" not in sl.session_state:
        sl.session_state["knowledge_base"] = None

def save_uploadedfile(uploadedfile):
    """Salva o arquivo PDF enviado na pasta 'uploaded'."""
    with open(os.path.join("uploaded", uploadedfile.name), "wb") as f:
        f.write(uploadedfile.getbuffer())

def remove_files():
    """Remove arquivos PDF do diretório 'uploaded'."""
    path = os.path.join(os.getcwd(), 'uploaded')
    for file_name in os.listdir(path):
        file = os.path.join(path, file_name)
        if os.path.isfile(file) and file.endswith(".pdf"):
            print('Deleting file:', file)
            os.remove(file)

if __name__ == '__main__':
    
    # --- Sidebar (Upload e Processamento) ---
    with sl.sidebar:
        with sl.form("my-form", clear_on_submit=True):
            pdf_docs = sl.file_uploader(label="Faça o Upload do seu PDF:", accept_multiple_files=True, type=["pdf"])
            submitted = sl.form_submit_button("Processar")
            
        if submitted and pdf_docs:
            initialize_session_state()
            
            remove_files() 
            for pdf in pdf_docs:
                save_uploadedfile(pdf)
                
            with sl.spinner("Processando PDFs..."):
                sl.session_state.knowledge_base = extract_data()
            
            if sl.session_state.knowledge_base:
                remove_files() 
                alert = sl.success(body=f"Upload e Processamento concluídos com Sucesso!", icon="✅")
            else:
                 alert = sl.error(body=f"Erro no processamento do PDF.", icon="❌")
            
            time.sleep(3) 
            alert.empty()


    # --- Aplicação Principal (Chat) ---
    sl.header("Bem-vindo ao PDF Chat")
    
    llm=load_llm()
    prompt=load_prompt()
    
    query=sl.text_input(label='Faça uma pergunta sobre o PDF:')
    
    if query:
        if sl.session_state.get("knowledge_base") is None:
            alert = sl.warning("Por favor, realize o Upload e Processamento do(s) PDF(s) para iniciar o Chat.", icon="🚨")
            time.sleep(3)
            alert.empty()
        else:
            try:
                retriever = sl.session_state.knowledge_base.as_retriever()
                
                rag_chain = (
                    {"context": retriever | format_docs, "question": RunnablePassthrough()}
                    | prompt
                    | llm
                    | StrOutputParser()
                )
                
                with sl.spinner("Gerando resposta..."):
                    response=rag_chain.invoke(query)
                
                sl.write(response)
                
            except Exception as e:
                print(f"Ocorreu um erro durante a execução do RAG: {e}")
                sl.error("Ocorreu um erro ao processar sua pergunta. Verifique sua chave de API ou se a API do Gemini está com problemas.", icon="❌")