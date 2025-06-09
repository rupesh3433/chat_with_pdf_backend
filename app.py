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
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

app = Flask(__name__)

# Enhanced CORS configuration
CORS(app, 
     resources={r"/*": {"origins": "*"}},
     methods=['GET', 'POST', 'DELETE', 'OPTIONS'],
     allow_headers=['Content-Type', 'Authorization'],
     supports_credentials=True)

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# Set environment variables for OpenRouter
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

class PDFChatSession:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.documents = []
        self.vectorstore = None
        self.chain = None
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
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

            doc = Document(
                page_content=text,
                metadata={"source": filename, "type": "pdf"}
            )
            self.documents.append(doc)
            print(f"Added document: {filename}, total documents: {len(self.documents)}")
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
        
        # Split documents into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", " ", ""]
        )
        splits = text_splitter.split_documents(self.documents)
        print(f"Created {len(splits)} text chunks")
        
        try:
            print("Setting up FAISS vectorstore...")
            
            if self.vectorstore:
                self.vectorstore.add_documents(splits)
                print("Added documents to existing vectorstore")
            else:
                self.vectorstore = FAISS.from_documents(splits, self.embeddings)
                print("Created new FAISS vectorstore")

            print("Setting up OpenRouter LLM...")
            
            # Use OpenRouter API with DeepSeek model (same as first code)
            llm = ChatOpenAI(
                model_name="deepseek/deepseek-chat-v3-0324:free",
                temperature=0.5,
                max_tokens=512,
                openai_api_base=os.environ["OPENAI_API_BASE"],
                openai_api_key=os.environ["OPENAI_API_KEY"]
            )
            
            self.chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
                memory=self.memory,
                return_source_documents=True,
                verbose=False
            )
            print("ConversationalRetrievalChain created successfully")
                
        except Exception as e:
            print(f"Error in _update_vectorstore: {str(e)}")
            import traceback
            traceback.print_exc()
            self.vectorstore = None
            self.chain = None

    def chat(self, question: str) -> Dict[str, Any]:
        print(f"Chat called with question: {question}")
        print(f"Chain exists: {self.chain is not None}")
        print(f"Vectorstore exists: {self.vectorstore is not None}")
        print(f"Number of documents: {len(self.documents)}")
        
        if not self.vectorstore:
            if len(self.documents) == 0:
                return {
                    "answer": "Please upload at least one PDF document first.",
                    "sources": []
                }
            else:
                return {
                    "answer": "There was an issue setting up the chat system. Please try uploading the document again.",
                    "sources": []
                }

        try:
            if self.chain:
                # Use the invoke method
                result = self.chain.invoke({"question": question})
                sources = []

                if "source_documents" in result:
                    for doc in result["source_documents"]:
                        sources.append({
                            "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                            "source": doc.metadata.get("source", "Unknown")
                        })

                return {
                    "answer": result["answer"],
                    "sources": sources
                }
            else:
                # Fallback: Just do similarity search and return relevant chunks
                docs = self.vectorstore.similarity_search(question, k=3)
                
                sources = []
                answer_parts = []
                
                for doc in docs:
                    sources.append({
                        "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                        "source": doc.metadata.get("source", "Unknown")
                    })
                    answer_parts.append(doc.page_content[:300])
                
                # Create a simple answer from the retrieved documents
                answer = f"Based on the documents, here are the most relevant sections for your question:\n\n"
                for i, part in enumerate(answer_parts, 1):
                    answer += f"{i}. {part[:200]}...\n\n"
                
                return {
                    "answer": answer,
                    "sources": sources
                }

        except Exception as e:
            print(f"Error in chat: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "answer": f"Error processing your question: {str(e)}",
                "sources": []
            }

    def get_session_info(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "total_documents": len(self.documents),
            "has_vectorstore": self.vectorstore is not None,
            "has_chain": self.chain is not None,
            "document_names": [doc.metadata.get("source", "Unknown") for doc in self.documents]
        }

# In-memory session store
sessions: Dict[str, PDFChatSession] = {}

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "active_sessions": len(sessions),
        "timestamp": datetime.now().isoformat()
    })

@app.route('/', methods=['GET'])
def root():
    return jsonify({'message': 'PDF Chat API with Sessions is running'}), 200

@app.route('/create-session', methods=['POST', 'OPTIONS'])
def create_session():
    if request.method == 'OPTIONS':
        return jsonify({'message': 'OK'}), 200
        
    session_id = str(uuid.uuid4())
    sessions[session_id] = PDFChatSession(session_id)
    print(f"Created session: {session_id}")
    return jsonify({
        "session_id": session_id,
        "message": "Session created successfully"
    })

@app.route('/upload-pdf', methods=['POST', 'OPTIONS'])
def upload_pdf():
    if request.method == 'OPTIONS':
        return jsonify({'message': 'OK'}), 200
        
    session_id = request.form.get('session_id')
    print(f"Upload request for session: {session_id}")

    if not session_id or session_id not in sessions:
        print(f"Invalid session ID: {session_id}")
        return jsonify({"error": "Invalid session ID"}), 400

    if 'pdf' not in request.files:
        return jsonify({"error": "No PDF file provided"}), 400

    pdf_file = request.files['pdf']

    if pdf_file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if not pdf_file.filename.lower().endswith('.pdf'):
        return jsonify({"error": "Only PDF files are allowed"}), 400

    try:
        pdf_content = pdf_file.read()
        session = sessions[session_id]
        success = session.add_pdf(pdf_content, pdf_file.filename)

        if success:
            print(f"Successfully uploaded PDF: {pdf_file.filename}")
            return jsonify({
                "message": f"PDF '{pdf_file.filename}' uploaded and processed successfully",
                "session_info": session.get_session_info()
            })
        else:
            print(f"Failed to process PDF: {pdf_file.filename}")
            return jsonify({"error": "Failed to process PDF"}), 400

    except Exception as e:
        print(f"Exception during upload: {str(e)}")
        return jsonify({"error": f"Error uploading PDF: {str(e)}"}), 500

@app.route('/chat', methods=['POST', 'OPTIONS'])
def chat():
    if request.method == 'OPTIONS':
        return jsonify({'message': 'OK'}), 200
        
    data = request.get_json()
    session_id = data.get('session_id')
    question = data.get('question')
    
    print(f"Chat request - Session: {session_id}, Question: {question}")

    if not session_id or session_id not in sessions:
        print(f"Invalid session ID in chat: {session_id}")
        return jsonify({"error": "Invalid session ID"}), 400

    if not question:
        return jsonify({"error": "Question is required"}), 400

    session = sessions[session_id]
    result = session.chat(question)
    return jsonify(result)

@app.route('/session-info/<session_id>', methods=['GET', 'OPTIONS'])
def get_session_info(session_id):
    if request.method == 'OPTIONS':
        return jsonify({'message': 'OK'}), 200
        
    if session_id not in sessions:
        return jsonify({"error": "Session not found"}), 404
    
    session = sessions[session_id]
    return jsonify(session.get_session_info())

@app.route('/clear-session/<session_id>', methods=['DELETE', 'OPTIONS'])
def clear_session(session_id):
    if request.method == 'OPTIONS':
        return jsonify({'message': 'OK'}), 200
        
    if session_id in sessions:
        del sessions[session_id]
        return jsonify({"message": "Session cleared successfully"})

    return jsonify({"error": "Session not found"}), 404

@app.route('/list-sessions', methods=['GET', 'OPTIONS'])
def list_sessions():
    if request.method == 'OPTIONS':
        return jsonify({'message': 'OK'}), 200
        
    session_list = []
    for session_id, session in sessions.items():
        session_list.append(session.get_session_info())
    
    return jsonify({
        "sessions": session_list,
        "total_sessions": len(sessions)
    })

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("Starting Flask PDF Chat Server with OpenRouter API...")
    print("Using DeepSeek model via OpenRouter")
    print(f"API Key configured: {'Yes' if os.getenv('OPENAI_API_KEY') else 'No'}")
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)