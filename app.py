from flask import Flask, request, jsonify
from flask_cors import CORS
import io
import os
from dotenv import load_dotenv
from typing import List, Dict, Any
import uuid
from datetime import datetime

# Load environment variables from .env
load_dotenv()

# PDF processing
from PyPDF2 import PdfReader

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

class PDFChatSession:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.documents = []
        self.vectorstore = None
        self.chain = None
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    def add_pdf(self, pdf_content: bytes, filename: str) -> bool:
        try:
            pdf_reader = PdfReader(io.BytesIO(pdf_content))
            text = ""
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text += f"\n[Page {page_num + 1}]\n{page_text}"

            if not text.strip():
                print(f"No text extracted from PDF: {filename}")
                return False

            doc = Document(page_content=text, metadata={"source": filename, "type": "pdf"})
            self.documents.append(doc)
            print(f"Added document: {filename}, total documents: {len(self.documents)})")
            self._update_vectorstore()
            return True
        except Exception as e:
            print(f"Error processing PDF {filename}: {str(e)}")
            return False

    def _update_vectorstore(self):
        if not self.documents:
            print("No documents to process")
            return

        print(f"Updating vectorstore with {len(self.documents)} documents")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        splits = text_splitter.split_documents(self.documents)

        try:
            if self.vectorstore:
                self.vectorstore.add_documents(splits)
                print("Added documents to existing vectorstore")
            else:
                self.vectorstore = FAISS.from_documents(splits, self.embeddings)
                print("Created new FAISS vectorstore")

            llm = ChatOpenAI(
                model_name="deepseek/deepseek-chat-v3-0324:free",
                temperature=0.5,
                max_tokens=512,
                openai_api_base=os.environ["OPENAI_API_BASE"],
                openai_api_key=os.environ["OPENAI_API_KEY"]
            )

            base_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
                return_source_documents=True
            )

            self.chain = RunnableWithMessageHistory(
                base_chain,
                lambda session_id: InMemoryChatMessageHistory(),
                input_messages_key="question",
                history_messages_key="chat_history"
            )
            print("RunnableWithMessageHistory chain created successfully")

        except Exception as e:
            print(f"Error in _update_vectorstore: {str(e)}")
            self.vectorstore = None
            self.chain = None

    def chat(self, question: str) -> Dict[str, Any]:
        if not self.chain or not self.vectorstore:
            return {"answer": "Please upload a PDF document first.", "sources": []}

        try:
            result = self.chain.invoke(
                {"question": question},
                config={"configurable": {"session_id": self.session_id}}
            )
            sources = []
            for doc in result.get("source_documents", []):
                sources.append({
                    "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                    "source": doc.metadata.get("source", "Unknown")
                })
            return {"answer": result["answer"], "sources": sources}
        except Exception as e:
            print(f"Error in chat: {str(e)}")
            return {"answer": f"Error: {str(e)}", "sources": []}

    def get_session_info(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "total_documents": len(self.documents),
            "has_vectorstore": self.vectorstore is not None,
            "has_chain": self.chain is not None,
            "document_names": [doc.metadata.get("source", "Unknown") for doc in self.documents]
        }

sessions: Dict[str, PDFChatSession] = {}

@app.route('/create-session', methods=['POST'])
def create_session():
    session_id = str(uuid.uuid4())
    sessions[session_id] = PDFChatSession(session_id)
    return jsonify({"session_id": session_id})

@app.route('/upload-pdf', methods=['POST'])
def upload_pdf():
    session_id = request.form.get('session_id')
    if not session_id or session_id not in sessions:
        return jsonify({"error": "Invalid session ID"}), 400

    if 'pdf' not in request.files:
        return jsonify({"error": "No PDF file provided"}), 400

    pdf_file = request.files['pdf']
    if pdf_file.filename == '' or not pdf_file.filename.lower().endswith('.pdf'):
        return jsonify({"error": "Invalid PDF file"}), 400

    session = sessions[session_id]
    success = session.add_pdf(pdf_file.read(), pdf_file.filename)
    if not success:
        return jsonify({"error": "Failed to process PDF"}), 400

    return jsonify({"message": "PDF uploaded", "session_info": session.get_session_info()})

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    session_id = data.get('session_id')
    question = data.get('question')

    if not session_id or session_id not in sessions:
        return jsonify({"error": "Invalid session ID"}), 400
    if not question:
        return jsonify({"error": "Question is required"}), 400

    session = sessions[session_id]
    return jsonify(session.chat(question))

@app.route('/session-info/<session_id>', methods=['GET'])
def get_session_info(session_id):
    if session_id not in sessions:
        return jsonify({"error": "Session not found"}), 404
    return jsonify(sessions[session_id].get_session_info())

@app.route('/clear-session/<session_id>', methods=['DELETE'])
def clear_session(session_id):
    if session_id in sessions:
        del sessions[session_id]
        return jsonify({"message": "Session cleared"})
    return jsonify({"error": "Session not found"}), 404

@app.route('/list-sessions', methods=['GET'])
def list_sessions():
    return jsonify({
        "sessions": [s.get_session_info() for s in sessions.values()],
        "total_sessions": len(sessions)
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "OK", "active_sessions": len(sessions)})

if __name__ == '__main__':
    print("Starting server...")
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
