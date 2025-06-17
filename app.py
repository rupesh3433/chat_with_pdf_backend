from flask import Flask, request, jsonify, make_response
from flask_cors import CORS, cross_origin
import io
import os
import sys
import traceback
from dotenv import load_dotenv
from typing import List, Dict, Any
import uuid
from datetime import datetime
import logging
import time

# Configure logging for production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables from .env
load_dotenv()

try:
    # PDF processing
    from PyPDF2 import PdfReader
    logger.info("PyPDF2 imported successfully")
except ImportError as e:
    logger.error(f"Failed to import PyPDF2: {e}")
    sys.exit(1)

# Global embeddings instance (will be initialized lazily)
global_embeddings = None
EMBEDDINGS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

try:
    # Updated LangChain imports
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain.chains import ConversationalRetrievalChain
    from langchain.schema import Document
    from langchain_openai import ChatOpenAI
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.embeddings import HuggingFaceEmbeddings as HFEmbeddings
    logger.info("LangChain imports successful")
except ImportError as e:
    logger.error(f"Failed to import LangChain components: {e}")

app = Flask(__name__)

# Updated CORS configuration with Render URL
CORS(app, 
     origins=[
         "https://chatwithdocuments.vercel.app",
         "https://chatwithdocuments-backend.onrender.com",
         "http://localhost:3000",
         "http://localhost:5173",
         "http://127.0.0.1:5000"
     ],
     methods=['GET', 'POST', 'DELETE', 'OPTIONS', 'PUT', 'PATCH'],
     allow_headers=[
         'Content-Type', 
         'Authorization', 
         'X-Requested-With', 
         'Accept', 
         'Origin'
     ],
     supports_credentials=False
)

# Environment setup with error handling
try:
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        logger.warning("OPENAI_API_KEY not found in environment variables")
        openai_api_key = "dummy-key"
    
    os.environ["OPENAI_API_KEY"] = openai_api_key
    os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"
    logger.info("Environment variables configured")
except Exception as e:
    logger.error(f"Error setting up environment: {e}")

def get_embeddings():
    """Get embeddings instance with lazy initialization and memory management"""
    global global_embeddings
    
    if global_embeddings is None:
        logger.info("Initializing embeddings model...")
        start_time = time.time()
        
        try:
            # Use more memory-efficient initialization
            global_embeddings = HFEmbeddings(
                model_name=EMBEDDINGS_MODEL,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            # Test with small input to force initialization
            global_embeddings.embed_query("test")
            
            load_time = time.time() - start_time
            logger.info(f"Embeddings initialized in {load_time:.2f} seconds")
        except Exception as e:
            logger.error(f"Embeddings initialization failed: {str(e)}")
            global_embeddings = None
    
    return global_embeddings

class PDFChatSession:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.documents = []
        self.vectorstore = None
        self.chain = None
        self.chat_history = []
        self.created_at = datetime.now().isoformat()

    def add_pdf(self, pdf_content: bytes, filename: str) -> bool:
        try:
            logger.info(f"Processing PDF: {filename}")
            pdf_reader = PdfReader(io.BytesIO(pdf_content))
            text = ""
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += f"\n[Page {page_num + 1}]\n{page_text}"
                except Exception as e:
                    logger.warning(f"Failed to extract text from page {page_num + 1}: {e}")
                    continue

            if not text.strip():
                logger.warning(f"No text extracted from PDF: {filename}")
                return False

            doc = Document(
                page_content=text,
                metadata={
                    "source": filename, 
                    "type": "pdf", 
                    "timestamp": datetime.now().isoformat(),
                    "page_count": len(pdf_reader.pages)
                }
            )
            self.documents.append(doc)
            self._update_vectorstore()
            logger.info(f"Successfully processed PDF: {filename}")
            return True

        except Exception as e:
            logger.error(f"Error processing PDF {filename}: {str(e)}")
            return False

    def _update_vectorstore(self):
        if not self.documents:
            logger.warning("Cannot update vectorstore: no documents")
            return

        try:
            # Split documents first to minimize memory usage
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=80,
                separators=["\n\n", "\n", " ", ""]
            )
            splits = text_splitter.split_documents(self.documents)
            logger.info(f"Split documents into {len(splits)} chunks")

            # Get embeddings (lazy-loaded)
            embeddings = get_embeddings()
            if not embeddings:
                logger.error("Embeddings not available - cannot create vectorstore")
                return

            # Process in batches to reduce memory pressure
            if self.vectorstore:
                self.vectorstore.add_documents(splits)
            else:
                # Create vectorstore in smaller batches
                batch_size = 20
                for i in range(0, len(splits), batch_size):
                    batch = splits[i:i+batch_size]
                    if not self.vectorstore:
                        self.vectorstore = FAISS.from_documents(batch, embeddings)
                    else:
                        self.vectorstore.add_documents(batch)
                logger.info(f"Vectorstore created with {len(splits)} documents")

            # Only initialize LLM chain if we have valid API key
            if os.environ.get("OPENAI_API_KEY") and os.environ["OPENAI_API_KEY"] != "dummy-key":
                try:
                    llm = ChatOpenAI(
                        model_name="deepseek/deepseek-chat-v3-0324:free",
                        temperature=0.3,
                        max_tokens=512,
                        openai_api_base=os.environ["OPENAI_API_BASE"],
                        openai_api_key=os.environ["OPENAI_API_KEY"],
                        request_timeout=60
                    )

                    self.chain = ConversationalRetrievalChain.from_llm(
                        llm=llm,
                        retriever=self.vectorstore.as_retriever(
                            search_type="similarity",
                            search_kwargs={"k": 3}
                        ),
                        return_source_documents=True,
                        verbose=False,
                        max_tokens_limit=4000
                    )
                    logger.info("LLM chain initialized successfully")
                except Exception as e:
                    logger.error(f"Failed to initialize LLM chain: {e}")
                    self.chain = None
            else:
                logger.warning("No valid API key - running in basic mode")
                self.chain = None

        except Exception as e:
            logger.error(f"Error in _update_vectorstore: {str(e)}")
            self.vectorstore = None
            self.chain = None

    def chat(self, question: str) -> Dict[str, Any]:
        if not self.vectorstore:
            if not self.documents:
                return {
                    "answer": "Please upload at least one PDF document to start chatting.", 
                    "sources": [],
                    "error": "no_documents"
                }
            return {
                "answer": "Vectorstore setup failed. Please try re-uploading your documents.", 
                "sources": [],
                "error": "vectorstore_failed"
            }

        try:
            if self.chain:
                result = self.chain.invoke({
                    "question": question, 
                    "chat_history": self.chat_history
                })
                
                self.chat_history.append((question, result["answer"]))
                if len(self.chat_history) > 8:
                    self.chat_history = self.chat_history[-8:]
                
                sources = []
                for doc in result.get("source_documents", []):
                    content = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
                    sources.append({
                        "content": content,
                        "source": doc.metadata.get("source", "Unknown"),
                        "page": doc.metadata.get("page", "N/A")
                    })
                
                return {
                    "answer": result["answer"], 
                    "sources": sources,
                    "mode": "ai_enhanced"
                }
            else:
                # Fallback mode without LLM
                docs = self.vectorstore.similarity_search(question, k=2)
                sources = []
                for doc in docs:
                    content = doc.page_content[:250] + "..." if len(doc.page_content) > 250 else doc.page_content
                    sources.append({
                        "content": content,
                        "source": doc.metadata.get("source", "Unknown")
                    })
                
                answer = f"Relevant sections for '{question}':\n\n" + \
                        "\n\n".join([f"{i+1}. From {src['source']}:\n{src['content']}" for i, src in enumerate(sources)])
                
                return {
                    "answer": answer, 
                    "sources": sources,
                    "mode": "basic_search"
                }

        except Exception as e:
            logger.error(f"Error in chat: {str(e)}")
            return {
                "answer": f"I encountered an error: {str(e)}", 
                "sources": [],
                "error": "processing_error"
            }

    def get_session_info(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "total_documents": len(self.documents),
            "has_vectorstore": self.vectorstore is not None,
            "has_chain": self.chain is not None,
            "document_names": [doc.metadata.get("source", "Unknown") for doc in self.documents],
            "chat_history_length": len(self.chat_history),
            "created_at": self.created_at,
            "last_activity": datetime.now().isoformat()
        }

    def clear_memory(self):
        self.chat_history = []
        logger.info(f"Memory cleared for session: {self.session_id}")

# Global sessions dictionary with automatic cleanup
sessions: Dict[str, PDFChatSession] = {}

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy", 
        "active_sessions": len(sessions), 
        "timestamp": datetime.now().isoformat(), 
        "version": "3.0",
        "environment": "production"
    })

@app.route('/', methods=['GET'])
def root():
    return jsonify({
        'message': 'Optimized PDF Chat API', 
        'version': '3.0', 
        'status': 'active',
        'endpoints': {
            'create_session': '/create-session',
            'upload_pdf': '/upload-pdf',
            'chat': '/chat',
            'session_info': '/session-info/<session_id>',
            'clear_session': '/clear-session/<session_id>',
            'clear_memory': '/clear-memory/<session_id>',
            'list_sessions': '/list-sessions',
            'health': '/health'
        }
    }), 200

@app.route('/create-session', methods=['POST'])
def create_session():        
    try:
        logger.info("Creating new session")
        session_id = str(uuid.uuid4())
        session = PDFChatSession(session_id)
        sessions[session_id] = session
        
        logger.info(f"Session created: {session_id}")
        
        return jsonify({
            "session_id": session_id, 
            "message": "Session created", 
            "created_at": session.created_at,
            "status": "ready"
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to create session: {str(e)}")
        return jsonify({
            "error": "Failed to create session",
            "error_type": "session_creation_failed"
        }), 500

@app.route('/upload-pdf', methods=['POST'])
def upload_pdf():        
    try:
        session_id = request.form.get('session_id')
        if not session_id or session_id not in sessions:
            return jsonify({
                "error": "Invalid session ID",
                "error_type": "invalid_session"
            }), 400
            
        if 'pdf' not in request.files:
            return jsonify({
                "error": "No PDF file provided",
                "error_type": "no_file"
            }), 400
        
        pdf_file = request.files['pdf']
        if not pdf_file.filename:
            return jsonify({
                "error": "No file selected",
                "error_type": "no_file_selected"
            }), 400
            
        if not pdf_file.filename.lower().endswith('.pdf'):
            return jsonify({
                "error": "Only PDF files allowed",
                "error_type": "invalid_file_type"
            }), 400
        
        # File size limit (5MB)
        pdf_content = pdf_file.read()
        if len(pdf_content) > 5 * 1024 * 1024:
            return jsonify({
                "error": "File too large (max 5MB)",
                "error_type": "file_too_large"
            }), 400
        
        session = sessions[session_id]
        success = session.add_pdf(pdf_content, pdf_file.filename)
        
        if success:
            return jsonify({
                "message": f"PDF processed successfully", 
                "session_info": session.get_session_info(),
                "status": "success"
            }), 200
        else:
            return jsonify({
                "error": "Failed to process PDF",
                "error_type": "pdf_processing_failed"
            }), 400
            
    except Exception as e:
        logger.error(f"Error uploading PDF: {str(e)}")
        return jsonify({
            "error": "Error processing PDF",
            "error_type": "upload_error"
        }), 500

@app.route('/chat', methods=['POST'])
def chat():        
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                "error": "No data provided",
                "error_type": "no_data"
            }), 400
            
        session_id = data.get('session_id')
        question = data.get('question', '').strip()
        
        if not session_id or session_id not in sessions:
            return jsonify({
                "error": "Invalid session ID",
                "error_type": "invalid_session"
            }), 400
            
        if not question:
            return jsonify({
                "error": "Question required",
                "error_type": "empty_question"
            }), 400
        
        session = sessions[session_id]
        result = session.chat(question)
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        return jsonify({
            "error": "Error processing request",
            "error_type": "chat_error"
        }), 500

@app.route('/session-info/<session_id>', methods=['GET'])
def get_session_info(session_id):        
    if session_id not in sessions:
        return jsonify({
            "error": "Session not found",
            "error_type": "session_not_found"
        }), 404
        
    return jsonify(sessions[session_id].get_session_info()), 200

@app.route('/clear-session/<session_id>', methods=['DELETE'])
def clear_session(session_id):        
    if session_id in sessions:
        del sessions[session_id]
        return jsonify({
            "message": "Session cleared",
            "status": "success"
        }), 200
    return jsonify({
        "error": "Session not found",
        "error_type": "session_not_found"
    }), 404

@app.route('/clear-memory/<session_id>', methods=['POST'])
def clear_memory(session_id):        
    if session_id not in sessions:
        return jsonify({
            "error": "Session not found",
            "error_type": "session_not_found"
        }), 404
        
    sessions[session_id].clear_memory()
    return jsonify({
        "message": "Memory cleared",
        "status": "success"
    }), 200

@app.route('/list-sessions', methods=['GET'])
def list_sessions():        
    session_list = [session.get_session_info() for session in sessions.values()]
    return jsonify({
        "sessions": session_list, 
        "total_sessions": len(sessions)
    }), 200

if __name__ == '__main__':
    logger.info("Starting optimized PDF Chat Server...")
    
    # Pre-initialize embeddings in main process
    logger.info("Preloading embeddings...")
    start = time.time()
    get_embeddings()
    logger.info(f"Embeddings preloaded in {time.time()-start:.2f}s")
    
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_ENV', 'production') == 'development'
    
    app.run(host='0.0.0.0', port=port, debug=debug_mode, threaded=True)