from flask import Flask, request, jsonify
import os
from utils import load_document, split_pages, create_vector_store, create_chain

app = Flask(__name__)
selected_embedding = "ollama"
vector_db = None
qa_chain = None