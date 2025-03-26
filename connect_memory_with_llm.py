import os

from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# step 1 setup LLM (Mistral with HugginFace)
HF_TOKEN = os.environ.get("HF_TOKEN")
huggingface_repo_id = "mistralai/Mistral-7B-Instruct-v0.3"


def load_llm(hugginface_repo_id):
    llm=HuggingFaceEndpoint(
        repo_id=hugginface_repo_id,
        temperature=0.5,
        model_kwargs={"token":HF_TOKEN,
                      "max_length":"512"}
    )
    return llm


# step 2 Create LLM with FAISS and create chain

custom_prompt_template = """
Use the pieces of information provided in the context to answer user's question.
if you dont know the answer, just say that you don't know, don't try to make up an answer.
Don't provide anything out of given context

Context: {context}
Question: {question}

Start the answer directly, No small talk please.
"""

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template,input_variables=["context","question"])
    return prompt

# Load database
db_faiss_path = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db=FAISS.load_local(db_faiss_path,embedding_model, allow_dangerous_deserialization=True)

# create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(huggingface_repo_id),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k':3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt': set_custom_prompt(custom_prompt_template)}
)


#now invoke with a single query

user_query = input("write query here :")
response = qa_chain.invoke({'query': user_query})
print("RESULT : ", response["result"])
print("SOURCE DOCUMENTS : ", response["source_documents"])