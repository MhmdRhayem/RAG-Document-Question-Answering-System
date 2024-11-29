from langchain.chat_models import ChatOpenAI
from langchain.llms import Ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader, PyPDFLoader
from sentence_transformers import SentenceTransformer
from langchain.embeddings.openai import OpenAIEmbeddings
import ollama
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferWindowMemory


def create_llm(model_type = "ollama"):
    if model_type == "ollama":
        return Ollama(model="phi3")
    else:
        return ChatOpenAI(model='gpt-3.5-turbo', temperature=0)

# Loading
def load_document(file_path):
    file_extension = file_path.split('.')[-1].lower()
    if file_extension == "txt":
        loader = TextLoader(file_path)
    elif file_extension == "pdf":
        loader = PyPDFLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")
    return loader.load()

# Splitting
def split_pages(pages):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    docs = text_splitter.split_documents(pages)
    return docs

# Embeddings
def hugging_face_embedding():    
    class MyEmbedding():
        def __init__(self):
            self.model = SentenceTransformer('all-MiniLM-L6-v2', trust_remote_code=True, truncate_dim=384)
                
        def embed_documents(self, docs):
            return [self.model.encode(doc).tolist() for doc in docs]
        
        def embed_query(self, query):
            return self.model.encode(query).tolist()