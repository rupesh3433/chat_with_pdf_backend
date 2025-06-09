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

# Updated LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import ChatMessageHistory  # Updated import

app = Flask(__name__)

CORS(app, 
     resources={r"/*": {"origins": ["https://chatwithdocuments.vercel.app", "http://localhost:5173"]}},
     methods=['GET', 'POST', 'DELETE', 'OPTIONS'],
     allow_headers=['Content-Type', 'Authorization'],
     supports_credentials=False  # keep False if you don't send credentials
)


# @app.after_request
# def after_request(response):
#     response.headers.add('Access-Control-Allow-Origin', '*')
#     response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
#     response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
#     return response

# Set OpenRouter API info
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

class PDFChatSession:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.documents = []
        self.vectorstore = None
        self.chain = None
        
        # Initialize chat history list (simpler approach)
        self.chat_history = []

        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    def add_pdf(self, pdf_content: bytes, filename: str) -> bool:
        """Add a PDF document to the session"""
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

            # Create document with metadata
            doc = Document(
                page_content=text, 
                metadata={"source": filename, "type": "pdf", "timestamp": datetime.now().isoformat()}
            )
            self.documents.append(doc)
            print(f"Added document: {filename}, total documents: {len(self.documents)}")
            
            # Update the vectorstore
            self._update_vectorstore()
            return True
            
        except Exception as e:
            print(f"Error processing PDF {filename}: {str(e)}")
            return False

    def _update_vectorstore(self):
        """Update or create the vectorstore with current documents"""
        if not self.documents:
            print("No documents to process")
            return

        print(f"Updating vectorstore with {len(self.documents)} documents")
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Increased chunk size for better context
            chunk_overlap=100,  # Increased overlap
            separators=["\n\n", "\n", " ", ""]
        )
        splits = text_splitter.split_documents(self.documents)
        print(f"Created {len(splits)} text chunks")

        try:
            # Create or update vectorstore
            if self.vectorstore:
                self.vectorstore.add_documents(splits)
                print("Added to existing vectorstore")
            else:
                self.vectorstore = FAISS.from_documents(splits, self.embeddings)
                print("Created new FAISS vectorstore")

            # Initialize the LLM
            llm = ChatOpenAI(
                model_name="deepseek/deepseek-chat-v3-0324:free",
                temperature=0.3,  # Slightly lower for more consistent responses
                max_tokens=1024,  # Increased token limit
                openai_api_base=os.environ["OPENAI_API_BASE"],
                openai_api_key=os.environ["OPENAI_API_KEY"]
            )

            # Create the conversational chain
            self.chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=self.vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 4}  # Retrieve more relevant chunks
                ),
                return_source_documents=True,
                verbose=False
            )
            print("ConversationalRetrievalChain created successfully")

        except Exception as e:
            print(f"Error in _update_vectorstore: {str(e)}")
            self.vectorstore = None
            self.chain = None

    def chat(self, question: str) -> Dict[str, Any]:
        """Process a chat question and return response with sources"""
        print(f"Chat called with question: {question}")
        
        if not self.vectorstore:
            if len(self.documents) == 0:
                return {
                    "answer": "Please upload at least one PDF document to start chatting.",
                    "sources": []
                }
            return {
                "answer": "Vectorstore setup failed. Please try re-uploading your documents.",
                "sources": []
            }

        try:
            if self.chain:
                # Use the conversational chain with chat history
                result = self.chain.invoke({
                    "question": question,
                    "chat_history": self.chat_history
                })
                
                # Update chat history
                self.chat_history.append((question, result["answer"]))
                
                # Keep only last 10 exchanges to manage memory
                if len(self.chat_history) > 10:
                    self.chat_history = self.chat_history[-10:]
                
                # Format sources
                sources = []
                for doc in result.get("source_documents", []):
                    content = doc.page_content
                    if len(content) > 400:
                        content = content[:400] + "..."
                    
                    sources.append({
                        "content": content,
                        "source": doc.metadata.get("source", "Unknown"),
                        "type": doc.metadata.get("type", "unknown")
                    })
                
                return {
                    "answer": result["answer"],
                    "sources": sources
                }
            else:
                # Fallback to similarity search
                docs = self.vectorstore.similarity_search(question, k=3)
                sources = []
                for doc in docs:
                    content = doc.page_content
                    if len(content) > 300:
                        content = content[:300] + "..."
                    
                    sources.append({
                        "content": content,
                        "source": doc.metadata.get("source", "Unknown"),
                        "type": doc.metadata.get("type", "unknown")
                    })
                
                answer = "Here are the most relevant sections from your documents:\n\n"
                for i, source in enumerate(sources, 1):
                    answer += f"{i}. From {source['source']}:\n{source['content']}\n\n"
                
                return {"answer": answer, "sources": sources}
                
        except Exception as e:
            print(f"Error in chat: {str(e)}")
            return {
                "answer": f"I encountered an error while processing your question: {str(e)}",
                "sources": []
            }

    def get_session_info(self) -> Dict[str, Any]:
        """Get information about the current session"""
        return {
            "session_id": self.session_id,
            "total_documents": len(self.documents),
            "has_vectorstore": self.vectorstore is not None,
            "has_chain": self.chain is not None,
            "document_names": [doc.metadata.get("source", "Unknown") for doc in self.documents],
            "chat_history_length": len(self.chat_history),
            "created_at": getattr(self, 'created_at', datetime.now().isoformat())
        }

    def clear_memory(self):
        """Clear the conversation memory"""
        self.chat_history = []
        print(f"Cleared memory for session {self.session_id}")

# Global session store
sessions: Dict[str, PDFChatSession] = {}

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "active_sessions": len(sessions),
        "timestamp": datetime.now().isoformat(),
        "version": "2.0"
    })

@app.route('/', methods=['GET'])
def root():
    """Root endpoint"""
    return jsonify({
        'message': 'PDF Chat API with Sessions is running',
        'version': '2.0',
        'endpoints': ['/create-session', '/upload-pdf', '/chat', '/session-info', '/clear-session', '/list-sessions']
    }), 200

@app.route('/create-session', methods=['POST', 'OPTIONS'])
def create_session():
    """Create a new chat session"""
    if request.method == 'OPTIONS':
        return jsonify({'message': 'OK'})
    
    session_id = str(uuid.uuid4())
    session = PDFChatSession(session_id)
    session.created_at = datetime.now().isoformat()
    sessions[session_id] = session
    
    return jsonify({
        "session_id": session_id,
        "message": "Session created successfully",
        "created_at": session.created_at
    })

@app.route('/upload-pdf', methods=['POST', 'OPTIONS'])
def upload_pdf():
    """Upload a PDF file to a session"""
    if request.method == 'OPTIONS':
        return jsonify({'message': 'OK'})
    
    session_id = request.form.get('session_id')
    if not session_id or session_id not in sessions:
        return jsonify({"error": "Invalid or expired session ID"}), 400
    
    if 'pdf' not in request.files:
        return jsonify({"error": "No PDF file provided"}), 400
    
    pdf_file = request.files['pdf']
    if pdf_file.filename == '' or not pdf_file.filename.lower().endswith('.pdf'):
        return jsonify({"error": "Only PDF files are allowed"}), 400

    try:
        pdf_content = pdf_file.read()
        session = sessions[session_id]
        success = session.add_pdf(pdf_content, pdf_file.filename)
        
        if success:
            return jsonify({
                "message": f"PDF '{pdf_file.filename}' uploaded and processed successfully",
                "session_info": session.get_session_info()
            })
        else:
            return jsonify({"error": "Failed to process PDF - no text could be extracted"}), 400
            
    except Exception as e:
        print(f"Upload error: {str(e)}")
        return jsonify({"error": f"Error uploading PDF: {str(e)}"}), 500

@app.route('/chat', methods=['POST', 'OPTIONS'])
def chat():
    """Chat with the uploaded documents"""
    if request.method == 'OPTIONS':
        return jsonify({'message': 'OK'})
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid JSON data"}), 400
            
        session_id = data.get('session_id')
        question = data.get('question', '').strip()
        
        if not session_id or session_id not in sessions:
            return jsonify({"error": "Invalid or expired session ID"}), 400
        
        if not question:
            return jsonify({"error": "Question is required and cannot be empty"}), 400
        
        session = sessions[session_id]
        response = session.chat(question)
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Chat error: {str(e)}")
        return jsonify({"error": f"Error processing chat: {str(e)}"}), 500

@app.route('/session-info/<session_id>', methods=['GET', 'OPTIONS'])
def get_session_info(session_id):
    """Get information about a specific session"""
    if request.method == 'OPTIONS':
        return jsonify({'message': 'OK'})
    
    if session_id not in sessions:
        return jsonify({"error": "Session not found"}), 404
    
    return jsonify(sessions[session_id].get_session_info())

@app.route('/clear-session/<session_id>', methods=['DELETE', 'OPTIONS'])
def clear_session(session_id):
    """Delete a session"""
    if request.method == 'OPTIONS':
        return jsonify({'message': 'OK'})
    
    if session_id in sessions:
        del sessions[session_id]
        return jsonify({"message": "Session cleared successfully"})
    
    return jsonify({"error": "Session not found"}), 404

@app.route('/clear-memory/<session_id>', methods=['POST', 'OPTIONS'])
def clear_memory(session_id):
    """Clear the conversation memory for a session"""
    if request.method == 'OPTIONS':
        return jsonify({'message': 'OK'})
    
    if session_id not in sessions:
        return jsonify({"error": "Session not found"}), 404
    
    sessions[session_id].clear_memory()
    return jsonify({"message": "Session memory cleared successfully"})

@app.route('/list-sessions', methods=['GET', 'OPTIONS'])
def list_sessions():
    """List all active sessions"""
    if request.method == 'OPTIONS':
        return jsonify({'message': 'OK'})
    
    session_list = []
    for session in sessions.values():
        info = session.get_session_info()
        session_list.append(info)
    
    return jsonify({
        "sessions": session_list,
        "total_sessions": len(sessions)
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("Starting Flask PDF Chat Server with OpenRouter API...")
    print("Updated with latest LangChain imports and improved functionality")
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)