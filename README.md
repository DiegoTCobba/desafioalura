# 🎓 Agente Virtual AcademiaX

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?logo=streamlit)
![LangChain](https://img.shields.io/badge/LangChain-0.2-green)
![Cohere](https://img.shields.io/badge/Cohere-command--a--03--2025-purple)
![FAISS](https://img.shields.io/badge/FAISS-CPU-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

Agente de IA que responde preguntas en lenguaje natural sobre los documentos oficiales del **Instituto de Educación Superior Online AcademiaX**, usando RAG (Retrieval-Augmented Generation).

---

## 📋 Descripción

AcademiaX Virtual Agent permite a estudiantes y afiliados consultar en lenguaje natural:

- Reglamentos y normas académicas
- Políticas de devoluciones y reembolsos
- Programas de becas y financiamiento
- Programa de afiliados comerciales
- Guía de uso de la plataforma virtual
- Código de ética e integridad académica
- Protección de datos personales y términos de uso

---

## 🏗️ Arquitectura

```
┌─────────────────────────────────────────────────────┐
│               Streamlit UI (app.py)                  │
│  ┌─────────────────┐       ┌────────────────────┐   │
│  │   Chat Area     │       │  Botones rápidos   │   │
│  └─────────────────┘       └────────────────────┘   │
└──────────────────────────┬──────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────┐
│              Filtro de relevancia (LLM)              │
│         ¿Es una pregunta académica/institucional?    │
└──────────┬───────────────────────────┬──────────────┘
           │ Sí                        │ No
┌──────────▼──────────────┐  ┌────────▼───────────────┐
│   RetrievalQA Chain     │  │  Mensaje fuera de      │
│   LangChain + Cohere    │  │  contexto              │
└──────────┬──────────────┘  └────────────────────────┘
           │
┌──────────▼──────────────────────────────────────────┐
│              FAISS VectorStore                       │
│       HuggingFace Embeddings (local, sin costo)      │
│         10 PDFs institucionales AcademiaX            │
└─────────────────────────────────────────────────────┘
```

**Flujo de procesamiento:**

1. Los 10 PDFs se cargan con `PyPDFLoader` y todas sus páginas se unen en un solo texto por documento — evita que tablas o secciones que cruzan páginas queden partidas.
2. El texto se divide con `RecursiveCharacterTextSplitter` (`chunk_size=4000`, `overlap=300`).
3. Cada fragmento se convierte en embedding con `sentence-transformers` (corre localmente) y se indexa en FAISS.
4. Ante una pregunta libre, un filtro LLM detecta si es relevante al dominio académico. Si no, el agente informa que no puede ayudar.
5. Si es relevante, se recuperan los `k=8` fragmentos más similares y se envían con la pregunta a Cohere usando un prompt personalizado que exige respuestas completas.

---

## 🛠️ Tecnologías

| Componente     | Tecnología                                       |
|----------------|--------------------------------------------------|
| UI             | Streamlit                                        |
| LLM            | Cohere `command-a-03-2025`                       |
| RAG Framework  | LangChain (classic + community)                  |
| Vector Store   | FAISS CPU                                        |
| Embeddings     | `sentence-transformers/all-MiniLM-L6-v2` (local) |
| PDF Loader     | PyPDF + LangChain                                |
| Variables env  | python-dotenv                                    |

---

## 📁 Estructura del proyecto

```
desafioalura/
├── app.py                  ← Aplicación principal (Streamlit + RAG)
├── diagnostico.py          ← Script de depuración y verificación
├── requirement.txt         ← Dependencias Python
├── .env.example            ← Plantilla de variables de entorno
├── .gitignore
├── README.md
└── Documentos/             ← PDFs institucionales fuente
    ├── 01_Reglamento_General_del_Estudiante.pdf
    ├── 02_Reglamento_Academico.pdf
    ├── 03_Politica_de_Devoluciones_y_Reembolsos.pdf
    ├── 04_Preguntas_Frecuentes_FAQ.pdf
    ├── 05_Guia_de_Uso_de_la_Plataforma.pdf
    ├── 06_Programa_de_Becas_y_Financiamiento.pdf
    ├── 07_Programa_de_Afiliados.pdf
    ├── 08_Terminos_y_Condiciones_de_Uso.pdf
    ├── 09_Politica_de_Proteccion_de_Datos_Personales.pdf
    └── 10_Codigo_de_Etica_Conducta_e_Integridad_Academica.pdf
```

---

## 🚀 Instalación local

### Requisitos previos

- Python 3.12+
- API key gratuita de Cohere → [dashboard.cohere.com](https://dashboard.cohere.com) → API Keys → Trial key

### Paso a paso

```bash
# 1. Clonar el repositorio
git clone https://github.com/DiegoTCobba/desafioalura.git
cd desafioalura

# 2. Crear y activar entorno virtual
python -m venv venv

# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

# 3. Instalar dependencias
pip install -r requirement.txt

# 4. Configurar la API key
copy .env.example .env   # Windows
cp .env.example .env     # Mac/Linux
# Editar .env y pegar tu COHERE_API_KEY

# 5. Ejecutar
streamlit run app.py
```

> La primera carga construye el índice FAISS (~30 segundos). Las siguientes son inmediatas.

### (Opcional) Verificar el sistema de búsqueda

```bash
python diagnostico.py
```

Muestra cuántos fragmentos se generan, cuáles recupera el sistema y las respuestas completas del modelo — útil para depurar.

---

## ⚙️ Configuración

Copia `.env.example` como `.env` y completa:

```
COHERE_API_KEY=tu_clave_aqui
```

Para deploy en **Streamlit Cloud**, agrega la clave en **Settings → Secrets**:

```
COHERE_API_KEY = "tu_clave_aqui"
```

---

## ☁️ Deploy en Streamlit Community Cloud

1. Sube el repositorio a GitHub (con la carpeta `Documentos/` y los PDFs).
2. Ingresa a [share.streamlit.io](https://share.streamlit.io).
3. Clic en **"New app"** y selecciona tu repositorio.
4. Configura: Branch `main` · Main file `app.py`.
5. En **Advanced settings → Secrets**, agrega tu `COHERE_API_KEY`.
6. Clic en **"Deploy"** ✅

**🔗 App desplegada:** [https://desafioalura-kakqk43agmzl2bau3ibgsr.streamlit.app](https://desafioalura-kakqk43agmzl2bau3ibgsr.streamlit.app)

---

## 📸 Evidencia del deploy

**Pantalla de inicio**

![Pantalla de inicio](evidencias/Captura%20de%20pantalla%202026-07-26%20233143.png)

**Respuesta sobre calificaciones**

![Calificaciones](evidencias/Captura%20de%20pantalla%202026-07-26%20230709.png)

**Respuesta sobre becas**

![Becas](evidencias/Captura%20de%20pantalla%202026-07-26%20230735.png)

---

## 💬 Ejemplos de uso

**Pregunta:** "Si saco 10.5 en un programa de especialización, ¿aprobaré?"

**Respuesta:** Según el Reglamento Académico (Art. 9°), la nota mínima aprobatoria para programas de especialización y diplomados es de **13/20** en cada módulo y en el proyecto integrador. Con 10.5, **no aprobarías**.

---

**Pregunta:** "¿Qué tipos de becas ofrece AcademiaX?"

**Respuesta:** AcademiaX ofrece:
- **Beca de Excelencia Académica** — promedio ≥ 18/20, cubre hasta el 50%.
- **Beca Socioeconómica** — previa evaluación, cubre entre 20% y 60%.
- **Beca por Convenio Institucional** — para colaboradores de empresas afiliadas.
- **Financiamiento en cuotas** — hasta 12 cuotas sin interés.

---

**Pregunta:** "¿Cuál es la política de devoluciones?"

**Respuesta:**
- Retiro dentro de los primeros 7 días: reembolso del **100%**.
- Retiro entre el día 8 y el día 15: reembolso del **50%**.
- Retiro después del día 15: **sin reembolso**.

---

**Pregunta:** "¿Quién ganó el último mundial?" *(fuera de contexto)*

**Respuesta:** Esa pregunta está fuera del alcance de mis documentos. Solo puedo ayudarte con temas relacionados a los reglamentos y políticas de AcademiaX.

---

## 👥 Créditos

Desarrollado como proyecto del **Desafío Alura LATAM — Agente de IA con RAG**.

Tecnologías: LangChain · Cohere · FAISS · Streamlit · HuggingFace · Python

