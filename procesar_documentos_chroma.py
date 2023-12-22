import os
import openai
import argparse
from langchain.document_loaders import PyPDFLoader
from langchain.schema import Document
#---------------
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from fastapi.responses import HTMLResponse
#---------------
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
from fastapi import FastAPI, Form
from fastapi import FastAPI, Request

import uvicorn

#-------------------
import shutil
from pydantic import BaseModel
from fastapi.templating import Jinja2Templates
#--------------------


# --------------------------------
#            API KEY 
# --------------------------------
# Se carga la clave de API de OpenAI desde un archivo .env

load_dotenv()  # Carga las variables de entorno del archivo .env
openai_api_key = os.getenv("OPENAI_API_KEY")
x1 = openai.api_key
embeddings = OpenAIEmbeddings(openai_api_key=x1)

CHROMA_PATH = "chroma_chatbot"  # Directorio donde se almacenará la base de datos
DATA_PATH = "Data"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


app = FastAPI()

templates = Jinja2Templates(directory="templates")


# Función para extraer el contenido del PDF y dividirlo en párrafos
def extract_pdf_content(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()
    paragraphs = [str(page.page_content) for page in pages]
    file_name = os.path.basename(pdf_path)  # Obtiene el nombre del archivo
    return paragraphs, file_name

# Función para generar objetos Document a partir de los párrafos
def embed_paragraphs(paragraphs, file_name):
    documents = [Document(page_content=paragraph, metadata={'file_name': file_name}) for paragraph in paragraphs]
    return documents

# Función para guardar documentos en una base de datos de Chroma
def save_to_chroma(documents):
    # Crea el directorio si no existe
    os.makedirs(CHROMA_PATH, exist_ok=True)

    # Crea y persiste la base de datos Chroma
    db = Chroma.from_documents(
        documents=documents,
        embedding=OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=openai_api_key),
        persist_directory=CHROMA_PATH
    )
    db.persist()

@app.get("/process_documents")
def process_documents():
    # Carga todos los PDF de la carpeta 'Data'
    pdf_files_list = [os.path.join("Data", file) for file in os.listdir("Data") if file.endswith('.pdf')]

    all_documents = []
    for pdf_path in pdf_files_list:
        paragraphs, file_name = extract_pdf_content(pdf_path)
        documents = embed_paragraphs(paragraphs, file_name)
        all_documents.extend(documents)

    save_to_chroma(all_documents)
    return {"status": "Procesamiento completado"}

def chat(data: str):
    # Your processing logic here
    # For example, you can assign a variable based on the form data
    result_variable = f"Processed Data: {data}"
    #return result_variable
    # Create CLI.
    # Esto es para poder escribir en consola
    #parser = argparse.ArgumentParser()
    #parser.add_argument("query_text", type=str, help="The query text.")
    # args = parser.parse_args()

    #----------------------------------
    #           QUERY 
    #----------------------------------
    #query_text = args.query_text
    query_text = result_variable

    #------------- Prepare the DB.
    # Lo primero que necesito es el path del API para autorizar el uso
    embedding_function = embeddings 


    # Aqui necesito crear un embbeding function que es la misma que utilizo para crear el DB
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Cuando se carga la base de datos ya se puede buscar por el chuck que mejor match el query
    # Se pasa el query_text com oun argument y se especifica el numero de resultados que deseamos
    # Best matches
    results = db.similarity_search_with_relevance_scores(query_text, k=3)

    # Los resultados vana ser tuplas
    # Antes de analizar los resultados se puede poner filtros
    # Si el resultado no existe o match es menor a 0.7
    if len(results) == 0 or results[0][1] < 0.7:
        print(f"Unable to find matching results.")
        return

    #----------------------------
    #       CREAR PROMPT
    #----------------------------
    # Todo esto lo que hace es juntar todos los chucks de data en una linea y la presenta 
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    #print(prompt)

    #----------------------------
    #       ESCOJER MODELO
    #----------------------------
    model = ChatOpenAI()
    response_text = model.predict(prompt)

    #----------------------------
    #     REFERENCES
    #----------------------------
    # Provide references back to your original material.
    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"{response_text}"

    return formatted_response


@app.post("/", response_class=HTMLResponse)
async def submit_form(request: Request, data: str = Form(...)):

    # Call another function and pass the form data
    result_variable = chat(data)

    return templates.TemplateResponse("form.html", {"request": request, "data": data, "response":result_variable})


@app.get("/", response_class=HTMLResponse)
async def show_form(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
