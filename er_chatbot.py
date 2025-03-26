import os

import streamlit as st

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())


db_faiss_path="vectorstore/db_faiss"
@st.cache_resource

def get_vectorstore():
    embedding_model=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db=FAISS.load_local(db_faiss_path, embedding_model, allow_dangerous_deserialization=True)
    return db


def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt


def load_llm(huggingface_repo_id, HF_TOKEN):
    llm=HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        model_kwargs={"token":HF_TOKEN,
                      "max_length":"512"}
    )
    return llm

def main():
    st.title("Ask EquityBot!")
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    
    prompt=st.chat_input("Pass your prompt here : ")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role':'user', 'content': prompt})


        custom_prompt_template = """
                Use the pieces of information provided in the context to answer user's question.
                if you dont know the answer, just say that you don't know, don't try to make up an answer.
                Don't provide anything out of given context

                Context: {context}
                Question: {question}

                Start the answer directly, No small talk please.
                """
        
        HUGGING_FACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
        HF_TOKEN=os.environ.get("HF_TOKEN")

        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")

            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(HUGGING_FACE_REPO_ID,HF_TOKEN),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k':3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt(custom_prompt_template)}
            )

            response=qa_chain.invoke({'query':prompt})

            result=response["result"]
            source_documents=response["source_documents"]
            result_to_show=result
            #+"\nSource Docs:\n"+str(source_documents)
            #response= "Hi, I am EquityBot!"
            st.chat_message('assistant').markdown(response)
            st.session_state.messages.append({'role':'assistant', 'content': result_to_show})
        
        except Exception as e:
            st.error(f"Error : {str(e)}")

if __name__ == "__main__":
    main()