import os
import glob
from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_cohere import ChatCohere
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
 
load_dotenv()
 
# ─────────────────────────────────────────────
#  CONFIGURACIÓN
# ─────────────────────────────────────────────
NOMBRE_INST = "AcademiaX"
NOMBRE_COMPLETO = "Instituto de Educación Superior Online AcademiaX"
CARPETA_DOCS = "Documentos"
ICONO = "🎓"
 
OPCIONES_RAPIDAS = {
    "📋 Reglamento del estudiante":      "¿Cuáles son los derechos y obligaciones del estudiante?",
    "📝 Calificaciones y evaluaciones":  "¿Cómo funciona el sistema de calificaciones y evaluaciones?",
    "💰 Devoluciones y reembolsos":      "¿Cuál es la política de devoluciones y reembolsos de matrícula?",
    "🎓 Becas y financiamiento":         "¿Qué tipos de becas y financiamiento ofrece AcademiaX y cuáles son los requisitos?",
    "🖥️ Uso de la plataforma virtual":   "¿Cómo accedo y uso la plataforma virtual de AcademiaX?",
    "🤝 Programa de afiliados":          "¿Cómo funciona el programa de afiliados y cuáles son las comisiones?",
    "🔒 Protección de datos":            "¿Cómo AcademiaX protege mis datos personales?",
    "⚖️ Código de ética":               "¿Cuáles son las normas de conducta e integridad académica?",
}
# ─────────────────────────────────────────────
 
st.set_page_config(page_title=f"Agente Virtual — {NOMBRE_INST}", page_icon=ICONO)
st.title(f"{ICONO} Agente Virtual — {NOMBRE_COMPLETO}")
 
MENSAJE_BIENVENIDA = (
    f"¡Hola! 👋 Soy el asistente virtual de **{NOMBRE_INST}**. "
    "Puedo responder preguntas sobre los reglamentos, becas, evaluaciones, "
    "devoluciones, la plataforma virtual, el código de ética, protección de datos "
    "y el programa de afiliados. ¿En qué puedo ayudarte hoy?"
)
 
MENSAJE_FUERA = (
    "Esa pregunta está fuera del alcance de mis documentos. 🙏 "
    f"Solo puedo ayudarte con temas relacionados a los reglamentos y políticas de {NOMBRE_INST}."
)
 
PLANTILLA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        f"Eres el asistente virtual oficial del {NOMBRE_COMPLETO}. "
        "Responde la pregunta del estudiante usando ÚNICAMENTE la información del "
        "siguiente contexto. Si la respuesta involucra una lista, tabla, porcentajes "
        "o pasos, incluye TODOS los elementos relevantes sin omitir ninguno. "
        "Antes de responder, revisa TODO el contexto para identificar cualquier "
        "sección relacionada con el tema. "
        "Si la información no está en el contexto, indícalo claramente sin inventar.\n\n"
        "Contexto:\n{context}\n\n"
        "Pregunta: {question}\n"
        "Respuesta completa:"
    ),
)
 
 
def encontrar_pdfs():
    return glob.glob(os.path.join(CARPETA_DOCS, "**", "*.pdf"), recursive=True)
 
 
@st.cache_resource
def cargar_llm():
    return ChatCohere(model="command-a-03-2025", temperature=0)
 
 
@st.cache_resource
def cargar_agente(rutas_pdfs):
    documentos_completos = []
    for ruta in rutas_pdfs:
        loader = PyPDFLoader(ruta)
        paginas = loader.load()
        texto_completo = "\n".join(p.page_content for p in paginas)
        documentos_completos.append(
            Document(page_content=texto_completo, metadata={"source": os.path.basename(ruta)})
        )
 
    splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=300)
    fragmentos = splitter.split_documents(documentos_completos)
 
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    base_vectorial = FAISS.from_documents(fragmentos, embeddings)
    retriever = base_vectorial.as_retriever(search_kwargs={"k": 8})
 
    return RetrievalQA.from_chain_type(
        llm=cargar_llm(),
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PLANTILLA_PROMPT},
    )
 
 
def es_pregunta_relevante(pregunta):
    instruccion = (
        "Responde ÚNICAMENTE con la palabra SI o la palabra NO, sin explicaciones.\n\n"
        f"¿La siguiente pregunta trata sobre temas académicos, administrativos o legales "
        f"de una institución educativa (reglamentos, calificaciones, becas, matrículas, "
        f"devoluciones, plataforma virtual, ética académica, datos personales, afiliados)?\n\n"
        f'Pregunta: "{pregunta}"'
    )
    respuesta = cargar_llm().invoke(instruccion)
    return respuesta.content.strip().upper().startswith("SI")
 
 
# ─── Validaciones ──────────────────────────────────────────────────────────
if not os.environ.get("COHERE_API_KEY"):
    st.error(
        "No encontré **COHERE_API_KEY**. Crea un archivo `.env` con: "
        "`COHERE_API_KEY=tu_clave` y ejecuta de nuevo `streamlit run app.py`."
    )
    st.stop()
 
pdfs = encontrar_pdfs()
if not pdfs:
    st.error(f"No hay PDFs en `{CARPETA_DOCS}/`. Coloca tus documentos ahí y reinicia.")
    st.stop()
 
with st.expander(f"📁 {len(pdfs)} documento(s) cargado(s)", expanded=False):
    for p in sorted(pdfs):
        st.markdown(f"- `{os.path.basename(p)}`")
 
with st.spinner("Preparando el agente (solo tarda la primera vez)..."):
    agente = cargar_agente(tuple(sorted(pdfs)))
 
# ─── Historial ─────────────────────────────────────────────────────────────
if "mensajes" not in st.session_state:
    st.session_state.mensajes = [{"role": "assistant", "content": MENSAJE_BIENVENIDA}]
 
for m in st.session_state.mensajes:
    with st.chat_message(m["role"]):
        st.write(m["content"])
        if m.get("fuentes"):
            with st.expander("Ver documentos consultados"):
                for doc in m["fuentes"]:
                    st.markdown(f"— *{doc.metadata.get('source', '?')}*")
 
# ─── Opciones rápidas ──────────────────────────────────────────────────────
pregunta_boton = None
st.write("O elige una opción rápida:")
columnas = st.columns(2)
for i, (etiqueta, pregunta_real) in enumerate(OPCIONES_RAPIDAS.items()):
    if columnas[i % 2].button(etiqueta, use_container_width=True, key=f"opcion_{i}"):
        pregunta_boton = pregunta_real
 
# ─── Entrada libre ─────────────────────────────────────────────────────────
pregunta_usuario = st.chat_input("Escribe tu pregunta...")
pregunta_final = pregunta_boton or pregunta_usuario
 
if pregunta_final:
    st.session_state.mensajes.append({"role": "user", "content": pregunta_final})
 
    if pregunta_boton or es_pregunta_relevante(pregunta_final):
        with st.spinner("Buscando en los documentos..."):
            resultado = agente.invoke({"query": pregunta_final})
        st.session_state.mensajes.append({
            "role": "assistant",
            "content": resultado["result"],
            "fuentes": resultado["source_documents"],
        })
    else:
        st.session_state.mensajes.append({"role": "assistant", "content": MENSAJE_FUERA})
 
    st.rerun()
