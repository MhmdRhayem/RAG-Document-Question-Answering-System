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
    