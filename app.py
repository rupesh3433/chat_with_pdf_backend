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

# Global embeddings instance
global_embeddings = None

try:
    # Updated LangChain imports
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain.chains import ConversationalRetrievalChain
    from langchain.schema import Document
    from langchain_openai import ChatOpenAI
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.chat_message_histories import ChatMessageHistory
    logger.info("LangChain imports successful")
    
    # Initialize global embeddings
    try:
        global_embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        logger.info("Global HuggingFace embeddings initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize global embeddings: {e}")
        global_embeddings = None

except ImportError as e:
    logger.error(f"Failed to import LangChain components: {e}")
    global_embeddings = None

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
        openai_api_key = "dummy-key"  # Allow app to start even without key
    
    os.environ["OPENAI_API_KEY"] = openai_api_key
    os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"
    logger.info("Environment variables configured")
except Exception as e:
    logger.error(f"Error setting up environment: {e}")

class PDFChatSession:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.documents = []
        self.vectorstore = None
        self.chain = None
        self.chat_history = []
        self.embeddings = global_embeddings  # Use global instance
        self.created_at = datetime.now().isoformat()

    def add_pdf(self, pdf_content: bytes, filename: str) -> bool:
        try:
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
        if not self.documents or not self.embeddings:
            logger.warning("Cannot update vectorstore: missing documents or embeddings")
            return

        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100,
                separators=["\n\n", "\n", " ", ""]
            )
            splits = text_splitter.split_documents(self.documents)
            logger.info(f"Split documents into {len(splits)} chunks")

            if self.vectorstore:
                self.vectorstore.add_documents(splits)
            else:
                self.vectorstore = FAISS.from_documents(splits, self.embeddings)

            # Only initialize LLM chain if we have API key
            if os.environ.get("OPENAI_API_KEY") and os.environ["OPENAI_API_KEY"] != "dummy-key":
                try:
                    llm = ChatOpenAI(
                        model_name="deepseek/deepseek-chat-v3-0324:free",
                        temperature=0.3,
                        max_tokens=1024,
                        openai_api_base=os.environ["OPENAI_API_BASE"],
                        openai_api_key=os.environ["OPENAI_API_KEY"]
                    )

                    self.chain = ConversationalRetrievalChain.from_llm(
                        llm=llm,
                        retriever=self.vectorstore.as_retriever(
                            search_type="similarity",
                            search_kwargs={"k": 4}
                        ),
                        return_source_documents=True,
                        verbose=False
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
                if len(self.chat_history) > 10:
                    self.chat_history = self.chat_history[-10:]
                
                sources = []
                for doc in result.get("source_documents", []):
                    content = doc.page_content[:400] + "..." if len(doc.page_content) > 400 else doc.page_content
                    sources.append({
                        "content": content,
                        "source": doc.metadata.get("source", "Unknown"),
                        "type": doc.metadata.get("type", "unknown"),
                        "page_count": doc.metadata.get("page_count", "Unknown")
                    })
                
                return {
                    "answer": result["answer"], 
                    "sources": sources,
                    "mode": "ai_enhanced"
                }
            else:
                # Fallback mode without LLM
                docs = self.vectorstore.similarity_search(question, k=3)
                sources = []
                for doc in docs:
                    content = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
                    sources.append({
                        "content": content,
                        "source": doc.metadata.get("source", "Unknown"),
                        "type": doc.metadata.get("type", "unknown")
                    })
                
                answer = f"Here are the most relevant sections from your documents for '{question}':\n\n" + \
                        "\n\n".join([f"{i+1}. From {src['source']}:\n{src['content']}" for i, src in enumerate(sources)])
                
                return {
                    "answer": answer, 
                    "sources": sources,
                    "mode": "basic_search"
                }

        except Exception as e:
            logger.error(f"Error in chat: {str(e)}")
            return {
                "answer": f"I encountered an error while processing your question: {str(e)}", 
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

# Global sessions dictionary
sessions: Dict[str, PDFChatSession] = {}

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy", 
        "active_sessions": len(sessions), 
        "timestamp": datetime.now().isoformat(), 
        "version": "2.1",
        "environment": "production"
    })

@app.route('/', methods=['GET'])
def root():
    return jsonify({
        'message': 'PDF Chat API with Sessions is running', 
        'version': '2.1', 
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
        
        logger.info(f"Session created successfully: {session_id}")
        
        response_data = {
            "session_id": session_id, 
            "message": "Session created successfully", 
            "created_at": session.created_at,
            "has_embeddings": session.embeddings is not None,
            "api_key_configured": os.environ.get("OPENAI_API_KEY", "").strip() not in ["", "dummy-key"],
            "status": "ready"
        }
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"Failed to create session: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "error": f"Failed to create session: {str(e)}",
            "error_type": "session_creation_failed"
        }), 500

@app.route('/upload-pdf', methods=['POST'])
def upload_pdf():        
    try:
        session_id = request.form.get('session_id')
        logger.info(f"PDF upload request for session: {session_id}")
        
        if not session_id or session_id not in sessions:
            return jsonify({
                "error": "Invalid or expired session ID",
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
                "error": "Only PDF files are allowed",
                "error_type": "invalid_file_type"
            }), 400
        
        # Check file size (limit to 10MB)
        pdf_content = pdf_file.read()
        if len(pdf_content) > 10 * 1024 * 1024:  # 10MB
            return jsonify({
                "error": "File too large. Maximum size is 10MB.",
                "error_type": "file_too_large"
            }), 400
        
        session = sessions[session_id]
        success = session.add_pdf(pdf_content, pdf_file.filename)
        
        if success:
            logger.info(f"PDF uploaded successfully: {pdf_file.filename}")
            return jsonify({
                "message": f"PDF '{pdf_file.filename}' uploaded and processed successfully", 
                "session_info": session.get_session_info(),
                "status": "success"
            }), 200
        else:
            return jsonify({
                "error": "Failed to process PDF - no text could be extracted",
                "error_type": "pdf_processing_failed"
            }), 400
            
    except Exception as e:
        logger.error(f"Error uploading PDF: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "error": f"Error uploading PDF: {str(e)}",
            "error_type": "upload_error"
        }), 500

@app.route('/chat', methods=['POST'])
def chat():        
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                "error": "No JSON data provided",
                "error_type": "no_data"
            }), 400
            
        session_id = data.get('session_id')
        question = data.get('question', '').strip()
        
        if not session_id or session_id not in sessions:
            return jsonify({
                "error": "Invalid or expired session ID",
                "error_type": "invalid_session"
            }), 400
            
        if not question:
            return jsonify({
                "error": "Question is required and cannot be empty",
                "error_type": "empty_question"
            }), 400
        
        session = sessions[session_id]
        result = session.chat(question)
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Error processing chat: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "error": f"Error processing chat: {str(e)}",
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
        logger.info(f"Session cleared: {session_id}")
        return jsonify({
            "message": "Session cleared successfully",
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
        "message": "Session memory cleared successfully",
        "status": "success"
    }), 200

@app.route('/list-sessions', methods=['GET'])
def list_sessions():        
    session_list = [session.get_session_info() for session in sessions.values()]
    return jsonify({
        "sessions": session_list, 
        "total_sessions": len(sessions),
        "timestamp": datetime.now().isoformat()
    }), 200

@app.errorhandler(404)
def not_found(error):
    response = jsonify({
        'error': 'Endpoint not found',
        'error_type': 'not_found',
        'available_endpoints': ['/health', '/create-session', '/upload-pdf', '/chat']
    })
    return response, 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {str(error)}")
    response = jsonify({
        'error': 'Internal server error',
        'error_type': 'internal_error'
    })
    return response, 500

@app.errorhandler(405)
def method_not_allowed(error):
    response = jsonify({
        'error': 'Method not allowed',
        'error_type': 'method_not_allowed'
    })
    return response, 405

if __name__ == '__main__':
    logger.info("Starting Flask PDF Chat Server with OpenRouter API...")
    
    # Check dependencies
    missing_deps = []
    try:
        import langchain
        import faiss
    except ImportError as e:
        missing_deps.append(str(e))
    
    if missing_deps:
        logger.warning(f"Some dependencies are missing: {missing_deps}")
        logger.warning("App will run in limited mode")
    
    # Use environment variable for port, defaulting to 5000
    port = int(os.environ.get('PORT', 5000))
    # In production, debug should be False
    debug_mode = os.environ.get('FLASK_ENV', 'production') == 'development'
    
    logger.info(f"Starting server on port {port}, debug={debug_mode}")
    logger.info(f"API Key configured: {os.environ.get('OPENAI_API_KEY', '').strip() not in ['', 'dummy-key']}")
    
    app.run(host='0.0.0.0', port=port, debug=debug_mode, threaded=True)