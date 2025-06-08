import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Enhanced CORS configuration
CORS(app, 
     resources={r"/*": {"origins": "*"}},  # Allow all origins for now
     methods=['GET', 'POST', 'DELETE', 'OPTIONS'],
     allow_headers=['Content-Type', 'Authorization'],
     supports_credentials=True)

# Alternative CORS setup for production
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# Configuration
UPLOAD_FOLDER = 'uploads'
VECTORSTORE_FOLDER = 'vectorstores'
ALLOWED_EXTENSIONS = {'pdf'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VECTORSTORE_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB limit

# Set environment variables for OpenRouter
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_documents(pdf_path):
    loader = PyPDFLoader(pdf_path)
    return loader.load_and_split()

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_documents(documents)

def create_vectorstore(chunks, filename):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    safe_filename = secure_filename(filename)
    vectorstore_path = os.path.join(VECTORSTORE_FOLDER, safe_filename)
    Chroma.from_documents(
        chunks, 
        embedding=embeddings,
        persist_directory=vectorstore_path
    )
    return vectorstore_path

def load_chat_model():
    return ChatOpenAI(
        model_name="deepseek/deepseek-chat-v3-0324:free",
        temperature=0.5,
        max_tokens=512,
        openai_api_base=os.environ["OPENAI_API_BASE"],
        openai_api_key=os.environ["OPENAI_API_KEY"]
    )

def load_vectorstore(filename):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    safe_filename = secure_filename(filename)
    vectorstore_path = os.path.join(VECTORSTORE_FOLDER, safe_filename)
    return Chroma(
        persist_directory=vectorstore_path,
        embedding_function=embeddings
    )

def build_qa_chain(llm, vectordb):
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'message': 'Server is running'}), 200

# Root endpoint
@app.route('/', methods=['GET'])
def root():
    return jsonify({'message': 'Chat with Documents API is running'}), 200

@app.route('/upload', methods=['POST', 'OPTIONS'])
def upload_file():
    if request.method == 'OPTIONS':
        return jsonify({'message': 'OK'}), 200
        
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        try:
            # Save file
            file.save(filepath)
            
            # Process the PDF and create vectorstore
            docs = load_documents(filepath)
            chunks = split_documents(docs)
            vectorstore_path = create_vectorstore(chunks, filename)
            
            return jsonify({
                'message': 'File successfully uploaded and processed',
                'filename': filename,
                'vectorstore': vectorstore_path
            }), 200
        except Exception as e:
            # Clean up if error occurs
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': f'Error processing file: {str(e)}'}), 500
    else:
        return jsonify({'error': 'Allowed file type is PDF (max 10MB)'}), 400

@app.route('/ask', methods=['POST', 'OPTIONS'])
def ask_question():
    if request.method == 'OPTIONS':
        return jsonify({'message': 'OK'}), 200
        
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    filename = data.get('filename')
    question = data.get('question')
    
    if not filename or not question:
        return jsonify({'error': 'Filename and question are required'}), 400
    
    try:
        # Load the vectorstore and LLM
        vectordb = load_vectorstore(filename)
        llm = load_chat_model()
        qa_chain = build_qa_chain(llm, vectordb)
        
        # Get the answer
        response = qa_chain.invoke(question)
        
        return jsonify({
            'question': question,
            'answer': response['result'],
            'sources': [doc.page_content for doc in response['source_documents']]
        }), 200
    except Exception as e:
        return jsonify({'error': f'Error processing question: {str(e)}'}), 500

@app.route('/list-documents', methods=['GET', 'OPTIONS'])
def list_documents():
    if request.method == 'OPTIONS':
        return jsonify({'message': 'OK'}), 200
        
    try:
        documents = []
        for filename in os.listdir(VECTORSTORE_FOLDER):
            dir_path = os.path.join(VECTORSTORE_FOLDER, filename)
            if os.path.isdir(dir_path):
                # Get original filename from uploads folder
                original_name = filename
                upload_path = os.path.join(UPLOAD_FOLDER, filename)
                if os.path.exists(upload_path):
                    size = os.path.getsize(upload_path)
                else:
                    size = 0
                
                documents.append({
                    'filename': original_name,
                    'size': size
                })
        return jsonify({'documents': documents}), 200
    except Exception as e:
        return jsonify({'error': f'Error listing documents: {str(e)}'}), 500

@app.route('/delete-document/<filename>', methods=['DELETE', 'OPTIONS'])
def delete_document(filename):
    if request.method == 'OPTIONS':
        return jsonify({'message': 'OK'}), 200
        
    try:
        # Clean up files
        safe_filename = secure_filename(filename)
        upload_path = os.path.join(UPLOAD_FOLDER, safe_filename)
        vectorstore_path = os.path.join(VECTORSTORE_FOLDER, safe_filename)
        
        if os.path.exists(upload_path):
            os.remove(upload_path)
            
        if os.path.exists(vectorstore_path):
            # Chroma stores data in a directory, so we need to remove the directory
            import shutil
            shutil.rmtree(vectorstore_path)
            
        return jsonify({'message': f'Document {filename} deleted successfully'}), 200
    except Exception as e:
        return jsonify({'error': f'Error deleting document: {str(e)}'}), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)