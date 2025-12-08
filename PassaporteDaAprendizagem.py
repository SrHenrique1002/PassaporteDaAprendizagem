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
os.makedirs(CURRICULUM_PDF_DIR, exist_ok=True) 
os.makedirs(FAISS_INDEX_DIR, exist_ok=True)

class Defasagem(BaseModel):
    """Esquema de saída estruturada para a análise do Boletim."""
    ano: str = Field(description="O ano letivo do estudante (ex: '7º ano'). Deve ser entre 6º e 9º ano.")
    bimestre: str = Field(description="O bimestre letivo da defasagem (ex: '3º bimestre'). Deve ser entre 1º e 4º.") 
    defasagem_foco: str = Field(description="O termo de busca ideal para a grade curricular (ex: 'Grade curricular completa de Matemática do 3º Bimestre').")
    motivo: str = Field(description="Breve justificativa baseada no boletim (ex: 'Média de 4.0 na unidade 3 em Álgebra').")


# ==============================================================================
# 1. FUNÇÕES DE CONFIGURAÇÃO E LLM (Mantidas)
# ==============================================================================

def load_prompt_rag():
    """Carrega o template de prompt RAG, com regra de separação para as abas."""
    prompt = """Você é um Analista Pedagógico e Avaliador Curricular de Matemática. 
Sua missão é criar um Pré-Questionário de 15 perguntas de diagnóstico para um aluno do **{ano}**, focado em identificar a defasagem exata no tópico: '{defasagem_foco}'.

# REGRAS DE GERAÇÃO:
1. **Foco Diagnóstico Amplo:** As 15 perguntas devem ser projetadas para testar os conceitos mais fundamentais e diversos presentes no 'Contexto Curricular Detalhado'.
2. **Contexto Exclusivo:** Use APENAS o 'Contexto Curricular Detalhado'.
3. **Formato de Saída OBRIGATÓRIO:**
    a. Comece com a **seção de Perguntas**.
    b. Após a última pergunta e antes de iniciar as Respostas, insira **EXATAMENTE** o delimitador: `---FIM_PERGUNTAS---`.
    c. Após o delimitador, inicie a seção de Respostas Detalhadas e Análise.

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
# 2. FUNÇÕES DE PROCESSAMENTO DE DATAS (Mantidas)
# ==============================================================================

def extract_year_from_filename(filename):
    """Extrai o ano letivo (6, 7, 8 ou 9) e o bimestre (1-4) do nome do arquivo."""
    cleaned_filename = filename.lower().replace('º', '').replace('°', '') 
    match_year = re.search(r'([6-9])\s*ano', cleaned_filename)
    year_key = None
    if match_year:
        year = match_year.group(1)
        year_key = f"{year}º ano"
        
    match_bimestre = re.search(r'([1-4])\s*bimestre', cleaned_filename)
    bimestre_key = None
    if match_bimestre:
        bimestre = match_bimestre.group(1)
        bimestre_key = f"B{bimestre}" 
        
    if year_key and bimestre_key:
        return f"{year_key} {bimestre_key}"
    return None

def get_sanitized_index_key(year_key: str) -> str:
    """Converte a chave legível para o nome de pasta FAISS seguro."""
    sanitized = year_key.lower().replace('º ano', 'ano').replace(' ', '_')
    return f"faiss_matematica_{sanitized}"

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
            continue

        index_name = get_sanitized_index_key(year_key) 
        index_path = os.path.join(FAISS_INDEX_DIR, index_name)
        
        if os.path.exists(index_path):
            indexed_years.add(year_key)
            continue 

        try:
            loader = PyPDFLoader(full_path)
            text_chunks = loader.load_and_split(text_splitter=text_splitter)
            
            vectorstore = FAISS.from_documents(documents=text_chunks, embedding=embeddings)
            vectorstore.save_local(index_path)
            indexed_years.add(year_key)
            
        except Exception as e:
            pass 
            
    if indexed_years:
        return f"Currículos indexados e prontos para consulta: {', '.join(sorted(list(indexed_years)))}."
    else:
        return "Nenhum currículo válido foi indexado. Verifique os PDFs e seus nomes."


def load_vector_store_by_year(year: str):
    """Carrega o Vector Store FAISS correto do disco."""
    embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004")
    index_name = get_sanitized_index_key(year) 
    index_path = os.path.join(FAISS_INDEX_DIR, index_name)
    
    if not os.path.exists(index_path):
        return None
        
    try:
        vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        return vectorstore
    except Exception:
        return None
        
def analyze_boletim(boletim_text, llm):
    """Analisa o texto do boletim usando LLM e retorna a defasagem estruturada."""
    
    llm_structured = llm.with_structured_output(Defasagem)
    
    analysis_prompt = ChatPromptTemplate.from_messages([
        ("system", 
        """Você é um Analista Pedagógico especialista em Matemática, anos 6º ao 9º. 
        Sua tarefa é analisar o desempenho do aluno no boletim fornecido e identificar:
        1. O **Ano** (6º ao 9º) e o **Bimestre** (1º ao 4º) em que o aluno obteve o pior desempenho em Matemática.
        2. O **termo de busca** ideal para a grade curricular desse período (`defasagem_foco`).

        # REGRAS DE INFERÊNCIA E SAÍDA:
        * Se o boletim **NÃO** detalhar unidades/tópicos (apenas médias bimestrais), use a frase de busca genérica: 'Grade curricular completa de Matemática do BIMESTRE X'.
        * Se o boletim **DETALHAR** o tópico (e a defasagem for clara), use o tópico específico (ex: 'Geometria Espacial').
        * Você DEVE retornar uma saída estruturada válida. Nunca retorne 'Não informado' nos campos `ano`, `bimestre` ou `defasagem_foco`.

        Instruções Finais: Retorne a saída estritamente no formato JSON, conforme o esquema Pydantic.

        """),
        ("user", "Analise o texto do boletim abaixo e extraia o ano e a defasagem de Matemática:\n\n{boletim_text}")])
    
    chain = (analysis_prompt | llm_structured)
    
    try:
        result = chain.invoke({"boletim_text": boletim_text}) 
        return result
    except Exception as e:
        sl.error(f"Erro ao analisar o boletim (LLM): {e}", icon="❌")
        return None

def extract_data_from_pdf(pdf_file):
    """Extrai texto do PDF do boletim."""
    try:
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
    """Inicializa o estado de sessão. Adiciona 'analysis_attempted'."""
    if "defasagem_data" not in sl.session_state: sl.session_state["defasagem_data"] = None 
    if "index_status" not in sl.session_state: sl.session_state["index_status"] = None
    if "response_rag" not in sl.session_state: sl.session_state["response_rag"] = None
    if "context_rag" not in sl.session_state: sl.session_state["context_rag"] = None
    if "ready_to_display" not in sl.session_state: sl.session_state["ready_to_display"] = False
    # NOVO: Indica se houve uma tentativa de análise clicando no botão
    if "analysis_attempted" not in sl.session_state: sl.session_state["analysis_attempted"] = False 
    
def save_uploadedfile(uploadedfile):
    """Salva o arquivo PDF enviado na pasta 'uploaded'."""
    with open(os.path.join("uploaded", uploadedfile.name), "wb") as f: f.write(uploadedfile.getbuffer())

# ==============================================================================
# 4. INTERFACE STREAMLIT E LÓGICA PRINCIPAL (Corrigida)
# ==============================================================================

if __name__ == '__main__':
    
    initialize_session_state()
    llm = load_llm()
    
    sl.markdown("### <span style='color: #4CAF50;'>🧠 Passaporte da Aprendizagem</span>", unsafe_allow_html=True)
    sl.subheader("Triagem Pedagógica e Geração de Questionário com Base no Boletim")
    
    # 1. Inicialização Automática (Verifica/Cria Índices FAISS)
    if sl.session_state.index_status is None:
        with sl.spinner("Verificando e indexando a Base Curricular local..."):
            sl.session_state.index_status = initialize_knowledge_base()

    # --- Sidebar ---
    with sl.sidebar:
        sl.markdown("## 📚 Configuração da Base RAG")
        sl.markdown(f"**Status da Base Curricular de Matemática:**")
        
        if "Currículos indexados" in sl.session_state.index_status:
            sl.success(sl.session_state.index_status, icon="✅")
        else:
            sl.error(sl.session_state.index_status, icon="❌")
            sl.markdown("---")
            sl.markdown(f"**Ação necessária:** Adicione os PDFs dos currículos (ex: `6ano_B1.pdf`) na pasta:")
            sl.code(CURRICULUM_PDF_DIR)

    sl.markdown("---") 

    if "Currículos indexados" not in sl.session_state.index_status:
        sl.warning("🚨 A base curricular não está pronta. Por favor, corrija a configuração na barra lateral para prosseguir.", icon="🚨")
        sl.stop()
        
    # --------------------------------------------------------------------------
    # ETAPA 1: Upload e Análise do Boletim
    # --------------------------------------------------------------------------
    
    with sl.container(border=True):
        sl.markdown("### 1️⃣ Upload e Identificação da Defasagem")
        
        boletim_pdf = sl.file_uploader(
            label='**Faça o upload do Boletim do Estudante (Formato PDF):**', 
            accept_multiple_files=False, 
            type=["pdf"],
            key="boletim_upload_key"
        )
        
        if boletim_pdf:
            # Sugestão: Limpar o estado de "análise tentada" se um novo PDF for carregado
            if sl.session_state.defasagem_data or sl.session_state.ready_to_display:
                sl.warning("Novo arquivo detectado. Clique em 'Analisar' para iniciar um novo diagnóstico.")
                sl.session_state["analysis_attempted"] = False

            if sl.button("Analisar Boletim e Identificar Foco de Estudo", type="primary", use_container_width=True, key="analyze_and_generate"):
                
                # AÇÃO 1: Registrar a tentativa de análise
                sl.session_state["analysis_attempted"] = True
                
                # LIMPEZA DO ESTADO:
                sl.session_state['ready_to_display'] = False
                sl.session_state['defasagem_data'] = None

                boletim_text = extract_data_from_pdf(boletim_pdf)
                
                if boletim_text:
                    sl.info("Boletim lido. Iniciando a análise para identificar a defasagem...", icon="🔍")
                    
                    with sl.spinner("Identificando o foco da defasagem em Matemática..."):
                        defasagem_result = analyze_boletim(boletim_text, llm)
                        sl.session_state.defasagem_data = defasagem_result
                        
                        # Se a análise falhar (defasagem_data continua None), o analysis_attempted=True garante que o erro será mostrado abaixo.
    
    # --------------------------------------------------------------------------
    # ETAPA 2: Processamento RAG
    # --------------------------------------------------------------------------
    
    if sl.session_state.defasagem_data:
        
        def_data = sl.session_state.defasagem_data
        
        sl.markdown("---") 
        
        with sl.container(border=True):
            sl.markdown("### 2️⃣ Busca Curricular e Geração do Questionário")
            sl.success(f"**Foco Encontrado:** **{def_data.ano}** | **{def_data.bimestre}** | Tópico: **{def_data.defasagem_foco}**", icon="💡")
            sl.caption(f"Motivo (Análise do Boletim): *{def_data.motivo}*")

            try:
                bimestre_match = re.search(r'[1-4]', def_data.bimestre)
                if not bimestre_match:
                    raise ValueError("Não foi possível extrair o número do bimestre (1-4).")
                    
                bimestre_num = bimestre_match.group(0) 
                key_busca = f"{def_data.ano} B{bimestre_num}" 
                
                with sl.spinner(f"Carregando base curricular específica: **{key_busca}**..."):
                    knowledge_base_specific = load_vector_store_by_year(key_busca)
                    
                if knowledge_base_specific is None:
                    sl.error(f"Falha no carregamento da base curricular de **{key_busca}**. O índice FAISS não existe ou está corrompido.", icon="❌")
                    sl.warning(f"Certifique-se de que o arquivo de currículo do **{def_data.ano}** e **{def_data.bimestre}** foi indexado corretamente na pasta `curriculos_base`.", icon="⚠️")
                    sl.stop() 
                    
                prompt_rag = load_prompt_rag()
                retriever = knowledge_base_specific.as_retriever(search_kwargs={"k": 10}) 
                
                with sl.spinner(f"Buscando conteúdo curricular e gerando questionário de diagnóstico para {def_data.defasagem_foco}..."):
                    
                    retrieved_docs = retriever.invoke(def_data.defasagem_foco)
                    context_str = format_docs(retrieved_docs)
                    
                    prompt_input = {
                        "context": context_str,
                        "defasagem_foco": def_data.defasagem_foco,
                        "ano": def_data.ano
                    }
                    
                    generation_chain = (prompt_rag | llm | StrOutputParser())
                    response = generation_chain.invoke(prompt_input) 
                    
                sl.success("Questionário de diagnóstico gerado com sucesso! Prossiga para a Etapa 3 abaixo.", icon="✅")
                
                sl.session_state['response_rag'] = response
                sl.session_state['context_rag'] = context_str
                sl.session_state['ready_to_display'] = True
                
            except Exception as e:
                sl.error(f"Ocorreu um erro fatal durante o RAG: {e}", icon="❌")
                sl.session_state['ready_to_display'] = False

    # [Trecho do Código da Etapa 3]
    # --------------------------------------------------------------------------
    # ETAPA 3: Exibição do Questionário (com Abas)
    # --------------------------------------------------------------------------

    if sl.session_state.get('ready_to_display', False):
        
        sl.markdown("---") 
        sl.markdown("### 3️⃣ Questionário de Diagnóstico Final")

        DELIMITER = "---FIM_PERGUNTAS---"
        response = sl.session_state['response_rag']
        def_data = sl.session_state.defasagem_data 
        
        if DELIMITER in response:
            perguntas_str, gabarito_str = response.split(DELIMITER, 1)
        else:
            perguntas_str = response
            gabarito_str = "⚠️ Falha ao separar perguntas e gabarito. Conteúdo completo na aba Questionário."
            sl.warning("O modelo LLM falhou ao inserir o delimitador.", icon="⚠️")

        tab1, tab2, tab3 = sl.tabs(["📋 Questionário", "🔍 Gabarito Pedagógico", "📚 Contexto Curricular (Auditoria)"])

        with tab1:
            sl.markdown("### Perguntas de Triagem Rápida")
            sl.markdown(perguntas_str)
            
        with tab2:
            sl.markdown("### Gabarito Detalhado e Análise")
            # Polimento UX: Aviso antes do gabarito
            sl.warning("⚠️ **ATENÇÃO:** Esta seção contém as respostas e análise pedagógica para uso do professor.", icon="🔥")
            sl.markdown(gabarito_str)

        with tab3:
            # Polimento UX: Contexto RAG escondido em expander
            sl.info(f"O questionário foi gerado usando o currículo do **{def_data.ano}** com foco em **{def_data.defasagem_foco}**.", icon="ℹ️")
            with sl.expander("Clique para ver os trechos do PDF utilizados (Contexto RAG)"):
                sl.markdown(sl.session_state['context_rag'])

            
    # CORREÇÃO APLICADA AQUI: Só mostra a mensagem de erro se o botão foi clicado (analysis_attempted = True)
    elif sl.session_state.get("analysis_attempted", False) and sl.session_state.defasagem_data is None:
         sl.error("❌ Não foi possível identificar a defasagem relevante no boletim. Tente um PDF com notas mais claras de Matemática ou recarregue a página para tentar novamente.", icon="⚠️")