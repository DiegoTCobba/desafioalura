import os
import glob
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_cohere import ChatCohere
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

load_dotenv()

CARPETA_DOCS = "documentos"
PDFS = sorted(glob.glob(os.path.join(CARPETA_DOCS, "**", "*.pdf"), recursive=True))

print("=" * 80)
print(f"PDFs encontrados: {len(PDFS)}")
for p in PDFS:
    print(f"  - {os.path.basename(p)}")

print("\n" + "=" * 80)
print("PASO 1: Cargando y dividiendo los documentos...")
print("=" * 80)

documentos_completos = []
for ruta in PDFS:
    loader = PyPDFLoader(ruta)
    paginas = loader.load()
    texto_completo = "\n".join(p.page_content for p in paginas)
    documentos_completos.append(
        Document(page_content=texto_completo, metadata={"source": os.path.basename(ruta)})
    )

splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=300)
fragmentos = splitter.split_documents(documentos_completos)

print(f"\nTotal de fragmentos generados: {len(fragmentos)}\n")
for i, frag in enumerate(fragmentos):
    origen = frag.metadata.get("source", "?")
    print(f"--- Fragmento {i} | origen: {origen} | {len(frag.page_content)} caracteres ---")
    print(frag.page_content[:200].replace("\n", " "))
    print("...")

print("\n" + "=" * 80)
print("PASO 2: Creando índice FAISS...")
print("=" * 80)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
base_vectorial = FAISS.from_documents(fragmentos, embeddings)
retriever = base_vectorial.as_retriever(search_kwargs={"k": 8})
print("Índice listo.\n")

PREGUNTAS_PRUEBA = [
    "¿Cuáles son los tipos de becas disponibles y sus requisitos?",
    "¿Cuál es la política de devoluciones y en qué plazos se realizan?",
    "¿Qué conductas constituyen una falta grave según el código de ética?",
    "¿Cómo funciona el programa de afiliados y cuáles son las comisiones?",
]

print("=" * 80)
print("PASO 3: Verificando recuperación (sin LLM)...")
print("=" * 80)
for pregunta in PREGUNTAS_PRUEBA:
    docs = retriever.invoke(pregunta)
    print(f"\n❓ {pregunta}")
    print(f"   Fragmentos recuperados: {len(docs)}")
    for d in docs:
        print(f"   - {d.metadata.get('source', '?')} ({len(d.page_content)} chars)")

print("\n" + "=" * 80)
print("PASO 4: Respuestas completas con el modelo de lenguaje...")
print("=" * 80)

llm = ChatCohere(model="command-a-03-2025", temperature=0)
plantilla = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "Eres el asistente virtual del Instituto de Educación Superior Online AcademiaX. "
        "Responde usando SOLO la información del contexto. Incluye TODOS los elementos "
        "relevantes (listas, tablas, porcentajes). Si no está en el contexto, dilo claramente.\n\n"
        "Contexto:\n{context}\n\n"
        "Pregunta: {question}\n"
        "Respuesta completa:"
    ),
)
agente = RetrievalQA.from_chain_type(
    llm=llm, retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": plantilla},
)

for pregunta in PREGUNTAS_PRUEBA:
    resultado = agente.invoke({"query": pregunta})
    print(f"\n❓ PREGUNTA: {pregunta}")
    print(f"📄 Docs usados: {[d.metadata.get('source','?') for d in resultado['source_documents']]}")
    print(f"💬 RESPUESTA: {resultado['result']}")
    print("-" * 80)
