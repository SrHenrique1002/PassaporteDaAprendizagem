import os
import time
import pandas as pd
import matplotlib.pyplot as plt
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# --- AJUSTE IMPORTANTE: Importando do seu arquivo específico ---
# O Python importa o arquivo 'PassaporteDaAprendizagem.py' (sem o .py)
try:
    from PassaporteDaAprendizagem import (
        initialize_knowledge_base, 
        load_vector_store_by_year, 
        analyze_boletim, 
        load_prompt_rag, 
        format_docs
    )
except ImportError as e:
    print("❌ Erro de Importação: Não foi possível encontrar o arquivo 'PassaporteDaAprendizagem.py'.")
    print("Certifique-se de que ele está na mesma pasta que este script.")
    exit()

# Configuração
load_dotenv()
# Nome exato da sua pasta com os 7 boletins
BOLETINS_DIR = "Boletim dos estudantes"  
RESULTADOS_CSV = "resultados_experimento.csv"

def run_experiment():
    print("🧪 INICIANDO EXPERIMENTO AUTOMATIZADO (Passaporte da Aprendizagem)...")
    print("-" * 60)

    # 1. Preparação
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    
    print("📂 Verificando base curricular em 'curriculos_base'...")
    # Essa função vai procurar a pasta 'curriculos_base' automaticamente conforme seu código original
    status_base = initialize_knowledge_base() 
    print(f"Status da Base: {status_base}")

    resultados = []

    # 2. Leitura da pasta de Boletins
    if not os.path.exists(BOLETINS_DIR):
        print(f"❌ A pasta '{BOLETINS_DIR}' não foi encontrada.")
        return

    arquivos_boletim = [f for f in os.listdir(BOLETINS_DIR) if f.lower().endswith(".pdf")]
    
    if not arquivos_boletim:
        print(f"❌ Nenhum PDF encontrado na pasta '{BOLETINS_DIR}'.")
        return

    print(f"🚀 Processando {len(arquivos_boletim)} boletins encontrados...\n")

    for i, arquivo in enumerate(arquivos_boletim):
        filepath = os.path.join(BOLETINS_DIR, arquivo)
        print(f"📄 [{i+1}/{len(arquivos_boletim)}] Analisando: {arquivo}...")
        
        metrics = {
            "arquivo": arquivo,
            "status": "Falha",
            "tempo_leitura": 0,
            "tempo_analise": 0,
            "tempo_geracao": 0,
            "tempo_total": 0,
            "defasagem_ano": "-",
            "defasagem_topico": "-"
        }

        start_total = time.time()

        try:
            # ETAPA A: Leitura do PDF
            t0 = time.time()
            loader = PyPDFLoader(filepath)
            docs = loader.load()
            texto_boletim = "\n\n".join(doc.page_content for doc in docs)
            metrics["tempo_leitura"] = time.time() - t0

            # ETAPA B: Análise da Defasagem (LLM)
            t1 = time.time()
            defasagem_result = analyze_boletim(texto_boletim, llm)
            metrics["tempo_analise"] = time.time() - t1

            if defasagem_result:
                metrics["defasagem_ano"] = defasagem_result.ano
                metrics["defasagem_topico"] = defasagem_result.defasagem_foco
                
                # Extração do número do bimestre para buscar na base
                import re
                bim_match = re.search(r'[1-4]', defasagem_result.bimestre)
                
                if bim_match:
                    bim_num = bim_match.group(0)
                    key_busca = f"{defasagem_result.ano} B{bim_num}"
                    
                    # ETAPA C: Busca Vetorial e Geração (RAG)
                    t2 = time.time()
                    vectorstore = load_vector_store_by_year(key_busca)
                    
                    if vectorstore:
                        retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
                        docs_rag = retriever.invoke(defasagem_result.defasagem_foco)
                        context_str = format_docs(docs_rag)
                        
                        # Geração final
                        prompt = load_prompt_rag()
                        chain = prompt | llm
                        # Invoca a chain para gerar o questionário
                        chain.invoke({
                            "context": context_str,
                            "defasagem_foco": defasagem_result.defasagem_foco,
                            "ano": defasagem_result.ano
                        })
                        
                        metrics["status"] = "Sucesso"
                    else:
                        metrics["status"] = f"Erro: Base {key_busca} não encontrada"
                    
                    metrics["tempo_geracao"] = time.time() - t2
                else:
                    metrics["status"] = "Erro: Bimestre não identificado"
            else:
                metrics["status"] = "Erro: LLM não retornou dados"

        except Exception as e:
            metrics["status"] = f"Erro Técnico"
            print(f"   ⚠️ Detalhe do erro: {e}")

        metrics["tempo_total"] = time.time() - start_total
        resultados.append(metrics)
        print(f"   ✅ Finalizado em {metrics['tempo_total']:.2f}s | Status: {metrics['status']}")

    # 3. Gerar Relatórios
    salvar_resultados(resultados)

def salvar_resultados(dados):
    if not dados: return

    df = pd.DataFrame(dados)
    df.to_csv(RESULTADOS_CSV, index=False)
    print(f"\n💾 Tabela salva em: {RESULTADOS_CSV}")

    # Exibe resumo no terminal
    print("\n📊 RESUMO DOS RESULTADOS:")
    print(df[["arquivo", "status", "defasagem_ano", "tempo_total"]].to_string(index=False))

    # Tenta gerar o gráfico
    try:
        df_sucesso = df[df["status"] == "Sucesso"]
        if not df_sucesso.empty:
            plt.figure(figsize=(10, 6))
            p1 = plt.bar(df_sucesso["arquivo"], df_sucesso["tempo_leitura"], label="Leitura PDF")
            p2 = plt.bar(df_sucesso["arquivo"], df_sucesso["tempo_analise"], bottom=df_sucesso["tempo_leitura"], label="Análise IA")
            p3 = plt.bar(df_sucesso["arquivo"], df_sucesso["tempo_geracao"], bottom=df_sucesso["tempo_leitura"]+df_sucesso["tempo_analise"], label="Geração Questões")
            
            plt.title("Performance por Boletim (Tempo em Segundos)")
            plt.legend()
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig("grafico_performance.png")
            print("📈 Gráfico gerado: grafico_performance.png")
    except Exception as e:
        print(f"Não foi possível gerar o gráfico: {e}")

if __name__ == "__main__":
    run_experiment()