import os
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from dotenv import load_dotenv
load_dotenv()

# Set environment variables for OpenRouter (DeepSeek V3 - Free)
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

def load_documents(pdf_path):
    loader = PyPDFLoader(pdf_path)
    return loader.load()

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(documents)

def create_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(chunks, embedding=embeddings)
    return vectorstore

def load_chat_model():
    return ChatOpenAI(
        model_name="deepseek/deepseek-chat-v3-0324:free",
        temperature=0.5,
        max_tokens=512,
        openai_api_base=os.environ["OPENAI_API_BASE"],
        openai_api_key=os.environ["OPENAI_API_KEY"]
    )

def build_qa_chain(llm, vectordb):
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

if __name__ == "__main__":
    file_path = "documents/sample.pdf"  # Replace this with your actual PDF file path
    print("[+] Loading and processing document...")
    docs = load_documents(file_path)
    chunks = split_documents(docs)
    vectordb = create_vectorstore(chunks)

    print("[+] Loading free DeepSeek-V3 model from OpenRouter...")
    llm = load_chat_model()
    qa_chain = build_qa_chain(llm, vectordb)

    print("\n💬 Chat is ready! Type your question (or 'exit'):\n")
    while True:
        query = input("You: ")
        if query.lower() == "exit":
            break
        response  = qa_chain.invoke(query)
        print("Bot:", response['result'])

        # Optional: print sources
        # for i, doc in enumerate(response['source_documents']):
        #     print(f"\n📄 Source {i+1}:")
        #     print(doc.page_content[:300], "...")
