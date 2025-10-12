import os
import streamlit as st

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langchain_groq import ChatGroq

# Optional local LLM (no external API)
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    from langchain_huggingface import HuggingFacePipeline
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False


## Uncomment the following files if you're not using pipenv as your virtual environment manager
#from dotenv import load_dotenv, find_dotenv
#load_dotenv(find_dotenv())


DB_FAISS_PATH="vectorstore/db_faiss"
@st.cache_resource
def get_vectorstore():
    embedding_model=HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        model_kwargs={'device':'cpu'}
    )
    db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db


def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt


def load_local_llm(model_name: str = "microsoft/DialoGPT-medium"):
    """Load a small local LLM via transformers; returns LangChain-compatible LLM or None."""
    if not TRANSFORMERS_AVAILABLE:
        return None
    try:
        tok = AutoTokenizer.from_pretrained(model_name)
        mdl = AutoModelForCausalLM.from_pretrained(model_name)
        gen = pipeline(
            task="text-generation",
            model=mdl,
            tokenizer=tok,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.5,
            pad_token_id=tok.eos_token_id,
        )
        return HuggingFacePipeline(pipeline=gen)
    except Exception:
        return None


def main():
    st.set_page_config(page_title="HealthConnect AI", page_icon="🩺", layout="wide")
    st.title("🩺 HealthConnect AI Assistant")
    st.subheader("Ask me about diseases, prevention, or vaccinations")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt=st.chat_input("Type your health question here...")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role':'user', 'content': prompt})

        CUSTOM_PROMPT_TEMPLATE = """
                Use the pieces of information provided in the context to answer user's question.
                If you dont know the answer, just say that you dont know, dont try to make up an answer. 
                Dont provide anything out of the given context

                Context: {context}
                Question: {question}

                Start the answer directly. No small talk please.
                """
        
        # Prefer local LLM; no external API required
        try: 
            vectorstore=get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")

            llm = load_local_llm()
            if llm is not None:
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=vectorstore.as_retriever(search_kwargs={'k':3}),
                    return_source_documents=True,
                    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
                )

                response=qa_chain.invoke({'query':prompt})
                result=response["result"]
                source_documents=response["source_documents"]
                result_to_show=result+"\nSource Docs:\n"+str(source_documents)
                st.chat_message('assistant').markdown(result_to_show)
                st.session_state.messages.append({'role':'assistant', 'content': result_to_show})
            else:
                # Fallback: no local LLM; return top docs text merged
                retriever = vectorstore.as_retriever(search_kwargs={'k':3})
                docs = retriever.get_relevant_documents(prompt)
                merged = "\n\n".join([d.page_content for d in docs]) if docs else "No relevant content found."
                result_to_show = merged
                st.chat_message('assistant').markdown(result_to_show)
                st.session_state.messages.append({'role':'assistant', 'content': result_to_show})

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()