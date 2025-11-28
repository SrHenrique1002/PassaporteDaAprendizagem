import streamlit as sl
import os
import time
import re
from pydantic import BaseModel, Field

# Importações LangChain e Google Gemini
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser 
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

# Garante que o diretório 'uploaded' exista e carrega variáveis de ambiente
os.makedirs("uploaded", exist_ok=True) 
load_dotenv()

# ==============================================================================
# 0. CONSTANTES E ESTRUTURAS DE DADOS
# ==============================================================================

# Definição dos diretórios
CURRICULUM_PDF_DIR = "curriculos_base" # Pasta onde seus PDFs de currículo devem estar
FAISS_INDEX_DIR = "faiss_indices"
os.makedirs(CURRICULUM_PDF_DIR, exist_ok=True) # Garante que a pasta de currículos exista
os.makedirs(FAISS_INDEX_DIR, exist_ok=True)

class Defasagem(BaseModel):
    """Esquema de saída estruturada para a análise do Boletim."""
    ano: str = Field(description="O ano letivo do estudante (ex: '7º ano'). Deve ser entre 6º e 9º ano.")
    defasagem_foco: str = Field(description="O tópico curricular de Matemática com a pior performance do aluno (ex: 'Geometria Espacial', 'Equações de 1º grau').")
    motivo: str = Field(description="Breve justificativa baseada no boletim (ex: 'Média de 4.0 na unidade 3 em Álgebra').")


# ==============================================================================
# 1. FUNÇÕES DE CONFIGURAÇÃO E LLM
# ==============================================================================

def load_prompt_rag():
    """Carrega o template de prompt RAG."""
    prompt = """Você é um Analista Pedagógico e Avaliador Curricular de Matemática. 
Sua missão é criar um Pré-Questionário de 5 perguntas de diagnóstico para um aluno do **{ano}**, focado em identificar a defasagem exata no tópico: '{defasagem_foco}'.

# REGRAS DE GERAÇÃO:
1. **Foco Diagnóstico:** As 5 perguntas devem ser projetadas para testar conceitos básicos e intermediários relacionados ao tópico e **identificar a raiz da dificuldade** do aluno. Não faça perguntas triviais ou excessivamente complexas.
2. **Contexto Exclusivo:** Use **APENAS** o 'Contexto Curricular Detalhado' recuperado para garantir a fidelidade ao programa de estudos do {ano}.
3. **Formato de Saída:** Formate o resultado em Markdown, incluindo 5 perguntas (discursivas curtas ou múltipla escolha) e, logo abaixo, as respostas detalhadas.

Contexto Curricular Detalhado (currículo do {ano}): {context}
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
# 2. FUNÇÕES DE PROCESSAMENTO DE DADOS (Indexação e Análise)
# ==============================================================================

def extract_year_from_filename(filename):
    """Extrai o ano letivo (6, 7, 8 ou 9) do nome do arquivo."""
    match = re.search(r'[6-9][º]?\s*ano|ano\s*[6-9]', filename, re.IGNORECASE)
    if match:
        year = re.search(r'[6-9]', match.group(0)).group(0)
        return f"{year}º ano"
    return None

def initialize_knowledge_base():
    """Verifica a pasta de currículos, indexa PDFs e retorna o status."""
    embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 512, chunk_overlap = 30, length_function = len, separators= ["\n\n", "\n", ".", " "]
    )
    
    indexed_years = set()
    pdf_files = [f for f in os.listdir(CURRICULUM_PDF_DIR) if f.endswith(".pdf")]
    
    if not pdf_files:
        return "Nenhum PDF de currículo encontrado no diretório 'curriculos_base'. Adicione os arquivos para iniciar."

    for pdf_name in pdf_files:
        full_path = os.path.join(CURRICULUM_PDF_DIR, pdf_name)
        year_key = extract_year_from_filename(pdf_name)
        
        if not year_key:
            sl.warning(f"Ignorando '{pdf_name}'. Não foi possível identificar o ano (6º-9º).", icon="⚠️")
            continue

        year_sanitized = year_key.replace('º ano', 'ano')
        index_path = os.path.join(FAISS_INDEX_DIR, f"faiss_matematica_{year_sanitized}")
        
        # 1. Verifica se o índice FAISS já existe
        if os.path.exists(index_path):
            indexed_years.add(year_key)
            continue 

        # 2. Se não existe, indexa e salva
        try:
            loader = PyPDFLoader(full_path)
            text_chunks = loader.load_and_split(text_splitter=text_splitter)
            
            vectorstore = FAISS.from_documents(documents=text_chunks, embedding=embeddings)
            vectorstore.save_local(index_path)
            indexed_years.add(year_key)
            
        except Exception as e:
            sl.error(f"Erro ao indexar o currículo do {year_key} ({pdf_name}): {e}", icon="❌")
            
    if indexed_years:
        return f"Currículos indexados e prontos para consulta: {', '.join(sorted(list(indexed_years)))}."
    else:
        return "Nenhum currículo válido foi indexado. Verifique os PDFs e seus nomes."


def load_vector_store_by_year(year: str):
    """Carrega o Vector Store FAISS correto do disco."""
    embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004")
    
    year_sanitized = year.replace('º ano', 'ano')
    index_path = os.path.join(FAISS_INDEX_DIR, f"faiss_matematica_{year_sanitized}")
    
    if not os.path.exists(index_path):
        return None
        
    try:
        # Usa allow_dangerous_deserialization=True para carregamento seguro do FAISS
        vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        return vectorstore
    except Exception:
        return None
def analyze_boletim(boletim_text, llm):
    """Analisa o texto do boletim usando LLM e retorna a defasagem estruturada."""
    
    # 1. Definir o LLM com o formato de saída estruturada
    # O PydanticOutputParser não é mais necessário, pois o LLM fará o parse.
    llm_structured = llm.with_structured_output(Defasagem)
    
    # 2. Definir o prompt simplificado (sem injeção de format_instructions)
    analysis_prompt = ChatPromptTemplate.from_messages([
        ("system", 
         """Você é um Analista Pedagógico especialista em Matemática, anos 6º ao 9º. 
         Sua tarefa é analisar o desempenho do aluno no boletim fornecido e identificar, na disciplina de Matemática, qual foi a unidade ou tópico curricular com o pior desempenho ou maior defasagem.
         Você deve retornar a saída estritamente no formato JSON, conforme o esquema Pydantic fornecido no template de saída.
         
         Instruções Específicas:
         1. Disciplina Foco: APENAS Matemática.
         2. Ano: Deve ser um valor entre 6º e 9º ano.
         """),
        ("user", "Analise o texto do boletim abaixo e extraia o ano e a defasagem de Matemática:\n\n{boletim_text}")
    ])
    
    # 3. Criar a cadeia (agora sem o parser, pois está no LLM)
    chain = (analysis_prompt | llm_structured)
    
    try:
        # A invocação retorna diretamente o objeto Pydantic (Defasagem)
        result = chain.invoke({"boletim_text": boletim_text}) 
        return result
    except Exception as e:
        # Esta exceção agora só deve ser acionada se o modelo não retornar JSON válido.
        sl.error(f"Erro ao analisar o boletim (LLM): {e}", icon="❌")
        return None

def extract_data_from_pdf(pdf_file):
    """Extrai texto do PDF do boletim."""
    try:
        # Nota: O 'uploaded' é usado para salvar o Boletim temporariamente.
        save_uploadedfile(pdf_file) 
        loader = PyPDFLoader(os.path.join('uploaded', pdf_file.name))
        
        documents = loader.load() 
        full_text = "\n\n".join(doc.page_content for doc in documents)
        
        return full_text
    except Exception as e:
        sl.error(f"Erro ao carregar o PDF do Boletim: {e}", icon="❌")
        return None
        
# ==============================================================================
# 3. FUNÇÕES AUXILIARES DE STREAMLIT
# ==============================================================================

def initialize_session_state():
    """Inicializa o estado de sessão."""
    if "defasagem_data" not in sl.session_state: sl.session_state["defasagem_data"] = None 
    if "index_status" not in sl.session_state: sl.session_state["index_status"] = None # Status da indexação base
    
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
# 4. INTERFACE STREAMLIT E LÓGICA PRINCIPAL
# ==============================================================================

if __name__ == '__main__':
    
    initialize_session_state()
    llm = load_llm()
    
    sl.title("🔬 Triagem Curricular por LLM (Diagnóstico de Boletim)")
    
    # 1. Inicialização Automática (Verifica/Cria Índices FAISS)
    with sl.spinner("Verificando e indexando a Base Curricular local..."):
        if sl.session_state.index_status is None:
            sl.session_state.index_status = initialize_knowledge_base()
            
    # --- Sidebar ---
    with sl.sidebar:
        sl.markdown("## 📚 Configuração da Base Curricular")
        sl.markdown(f"**Status da Base RAG:**")
        
        if "Currículos indexados" in sl.session_state.index_status:
            sl.success(sl.session_state.index_status, icon="✅")
        else:
            sl.error(sl.session_state.index_status, icon="❌")
            sl.markdown(f"**Ação necessária:** Adicione os PDFs dos currículos (ex: `Curriculo_7ano.pdf`) na pasta `{CURRICULUM_PDF_DIR}`.")
    
    # --- Aplicação Principal: Análise do Boletim e Geração ---
    
    sl.markdown("## 📈 Análise do Boletim e Geração do Diagnóstico")
    
    if "Currículos indexados" not in sl.session_state.index_status:
        # Se a indexação falhou, não permite o upload do Boletim
        sl.warning("🚨 A base curricular não está pronta. Por favor, corrija a configuração na barra lateral.", icon="🚨")
    else:
        
        # Uso da key única para evitar DuplicateWidgetID
        boletim_pdf = sl.file_uploader(
            label='**1. Upload do Boletim do Estudante (PDF):**', 
            accept_multiple_files=False, 
            type=["pdf"],
            key="boletim_upload_key" 
        )

        if boletim_pdf and sl.button("Analisar Boletim e Gerar Questionário", key="analyze_and_generate"):
            
            # A. Extrair texto do Boletim
            boletim_text = extract_data_from_pdf(boletim_pdf)
            
            if boletim_text:
                sl.info("Boletim lido. Iniciando a análise para identificar a defasagem...", icon="🔍")
                
                # B. Analisar o Boletim (LLM Chain 1)
                with sl.spinner("Identificando o foco da defasagem em Matemática..."):
                    defasagem_result = analyze_boletim(boletim_text, llm)
                    sl.session_state.defasagem_data = defasagem_result
                
                if sl.session_state.defasagem_data:
                    def_data = sl.session_state.defasagem_data
                    sl.success(f"Defasagem identificada no **{def_data.ano}**: **{def_data.defasagem_foco}**.", icon="💡")
                    sl.caption(f"Motivo (Análise do Boletim): {def_data.motivo}")
                    
                    # C. Mapeamento e Carregamento Dinâmico do Vector Store (Etapa 2)
                    with sl.spinner(f"Carregando base curricular específica do **{def_data.ano}**..."):
                        knowledge_base_specific = load_vector_store_by_year(def_data.ano)
                        
                    if knowledge_base_specific is None:
                        sl.error(f"Falha no carregamento da base curricular de {def_data.ano}. O índice FAISS não existe ou está corrompido.", icon="❌")
                        sl.stop() # Interrompe a execução
                        
                    # D. Preparar e Executar o RAG (LLM Chain 2) - ETAPA 3
                    try:
                        prompt_rag = load_prompt_rag()
                        retriever = knowledge_base_specific.as_retriever(search_kwargs={"k": 5})
                        
                        with sl.spinner(f"Buscando conteúdo curricular e gerando questionário de diagnóstico para {def_data.defasagem_foco}..."):
                            
                            # 1. Recupera o contexto relevante usando o tópico de defasagem como query
                            retrieved_docs = retriever.invoke(def_data.defasagem_foco)
                            context_str = format_docs(retrieved_docs)
                            
                            # 2. Monta o input para o prompt com todas as variáveis
                            prompt_input = {
                                "context": context_str,
                                "defasagem_foco": def_data.defasagem_foco,
                                "ano": def_data.ano
                            }
                            
                            # 3. Executa a cadeia de geração
                            generation_chain = (prompt_rag | llm | StrOutputParser())
                            response = generation_chain.invoke(prompt_input) 
                        
                        # Output
                        sl.subheader("2. Pré-Questionário de Diagnóstico Gerado:")
                        sl.markdown(response)
                        
                        # E. Auditoria e Transparência do RAG (Etapa 4 - NOVO)
                        with sl.expander("📚 Contexto Curricular Utilizado para Geração (Auditoria)"):
                            sl.markdown(context_str)
                        
                        sl.info(f"Questionário gerado com base no currículo do **{def_data.ano}** e no foco **{def_data.defasagem_foco}**.", icon="✅")
                        
                    except Exception as e:
                        sl.error(f"Ocorreu um erro ao processar o RAG: {e}", icon="❌")
                        
                else: # O 'else' pertence ao 'if sl.session_state.defasagem_data:' (falha na Etapa B)
                    sl.error("Não foi possível identificar a defasagem relevante no boletim.", icon="❌")