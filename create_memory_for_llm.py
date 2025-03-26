from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

data_path = "pdf/"
#Step1 load raw pdfs
def load_pdf_files(data):
    loader = DirectoryLoader(data,
                             glob='*.pdf',
                             loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

documents = load_pdf_files(data = data_path)
# print("Length of pdf pages : ", len(documents))

#step2 create chunks
def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200,
                                                   chunk_overlap=50)  #chunk_size try 300,400, overlap accordingly
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks



text_chunks = create_chunks(extracted_data=documents)
# print("Length of text chunks : ", len(text_chunks))


#step3 create vector embeddings

def get_embedding_model():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model

embedding_model = get_embedding_model()


#step4 store embedding in FAISS
db_faiss_path = "vectorstore/db_faiss/"
db = FAISS.from_documents(text_chunks, embedding_model)
db.save_local(db_faiss_path)
