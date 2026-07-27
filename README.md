# Agente Virtual — Instituto de Educación Superior Online AcademiaX

Agente de IA que responde preguntas en lenguaje natural sobre los documentos
oficiales de AcademiaX, usando RAG (Retrieval-Augmented Generation).

## Documentos fuente

El agente consulta 10 documentos PDF:

1. `01_Reglamento_General_del_Estudiante.pdf` — derechos, obligaciones y normas de la comunidad estudiantil.
2. `02_Reglamento_Academico.pdf` — evaluaciones, calificaciones, progresión y certificación.
3. `03_Politica_de_Devoluciones_y_Reembolsos.pdf` — plazos y condiciones para devolución de pagos.
4. `04_Preguntas_Frecuentes_FAQ.pdf` — respuestas oficiales sobre cursos, pagos, certificados y soporte.
5. `05_Guia_de_Uso_de_la_Plataforma.pdf` — navegación y herramientas de la plataforma virtual.
6. `06_Programa_de_Becas_y_Financiamiento.pdf` — criterios, requisitos y compromisos del becado.
7. `07_Programa_de_Afiliados.pdf` — condiciones, comisiones y obligaciones de los afiliados.
8. `08_Terminos_y_Condiciones_de_Uso.pdf` — contrato de prestación de servicios educativos en línea.
9. `09_Politica_de_Proteccion_de_Datos_Personales.pdf` — tratamiento de datos conforme a Ley N° 29733.
10. `10_Codigo_de_Etica_Conducta_e_Integridad_Academica.pdf` — valores, normas de convivencia e integridad.

## Tecnologías utilizadas

- **Python 3.12**
- **LangChain** (`langchain-classic`, `langchain-community`, `langchain-text-splitters`, `langchain-cohere`)
- **PyPDF** — lectura de documentos PDF
- **FAISS** — índice vectorial para búsqueda semántica
- **sentence-transformers** (`all-MiniLM-L6-v2`) — embeddings locales y gratuitos
- **Cohere** (`command-a-03-2025`) — modelo de lenguaje
- **Streamlit** — interfaz de chat web
- **python-dotenv** — manejo seguro de la API key

## Arquitectura

1. Cada PDF se carga con `PyPDFLoader` y **todas sus páginas se unen en un solo texto** antes de dividir — esto evita que tablas o secciones que cruzan páginas queden partidas.
2. El texto completo se divide con `RecursiveCharacterTextSplitter` (`chunk_size=4000`). Los 10 documentos generan un número reducido de fragmentos, cubriendo el contenido completo.
3. Cada fragmento se convierte en un embedding con `sentence-transformers` (corre localmente, sin costo) y se indexa en FAISS.
4. Ante una pregunta libre, un filtro con el LLM detecta si es relevante al dominio académico/institucional. Si no lo es, el agente informa que no puede ayudar.
5. Si la pregunta es relevante, se recuperan los `k=8` fragmentos más similares y se envían junto con la pregunta a Cohere, usando un prompt personalizado que exige respuestas completas.
6. Streamlit expone todo esto como una app de chat con saludo inicial y 8 botones de preguntas rápidas.

```
flowchart TD
    A[10 PDFs institucionales] --> B[Fragmentos + embeddings + FAISS]
    B --> C[Pregunta del usuario]
    C --> D{Es pregunta institucional?}
    D -- No --> E[Fuera de alcance]
    D -- Si --> F[FAISS k=8 + prompt + Cohere]
    F --> G[Respuesta completa]
```

## Cómo ejecutar el proyecto localmente

### Requisitos previos

- **Python 3.12** (`python --version`)
- **Git** instalado
- **API key gratuita de Cohere** — regístrate en [dashboard.cohere.com](https://dashboard.cohere.com) → API Keys → Trial key

### 1. Clonar el repositorio

```bash
git clone https://github.com/TU_USUARIO/agente-academiax.git
cd agente-academiax
```

### 2. Crear y activar el entorno virtual

```bash
python -m venv venv
```

- **Windows (PowerShell):** `venv\Scripts\Activate.ps1`
- **Windows (cmd):** `venv\Scripts\activate.bat`
- **Mac/Linux:** `source venv/bin/activate`

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

> La primera instalación puede tardar varios minutos (descarga `sentence-transformers` y `torch`).

### 4. Configurar la API key

```bash
# Windows
copy .env.example .env

# Mac/Linux
cp .env.example .env
```

Edita `.env` y reemplaza el valor:

```
COHERE_API_KEY=tu_clave_real_de_cohere
```

### 5. Ejecutar la aplicación

```bash
streamlit run app.py
```

Se abrirá en `http://localhost:8501`. La primera carga construye el índice FAISS (tarda ~30 segundos); las siguientes son inmediatas.

### (Opcional) Verificar el sistema de búsqueda

```bash
python diagnostico.py
```

Muestra cuántos fragmentos se generan, cuáles recupera el sistema para preguntas de prueba y las respuestas completas del modelo — útil para depurar antes de hacer cambios.

## Estructura del proyecto

```
agente-academiax/
├── app.py                  # Aplicación principal (Streamlit + RAG)
├── diagnostico.py          # Script de depuración y verificación
├── requirements.txt        # Dependencias Python
├── .env.example            # Plantilla de variables de entorno
├── .gitignore
└── documentos/             # PDFs institucionales fuente
    ├── 01_Reglamento_General_del_Estudiante.pdf
    ├── 02_Reglamento_Academico.pdf
    ├── ...
    └── 10_Codigo_de_Etica_Conducta_e_Integridad_Academica.pdf
```
