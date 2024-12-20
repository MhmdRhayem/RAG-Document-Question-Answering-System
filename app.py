from flask import Flask, request, jsonify
import os
from utils import load_document, split_pages, create_vector_store, create_chain, add_documents

app = Flask(__name__)
selected_embedding = "openai"
docs = None
vector_db = None
qa_chain = None

@app.route("/create_vector_store",methods = ["POST"])
def create_store():
    global vector_db
    selected_embedding = request.json["selected_embedding"]
    vector_db = create_vector_store(selected_embedding)
    print("Done Creating Vector Store")

@app.route("/upload", methods=["POST"])
def upload_document():
    global vector_db
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file:
        filename = file.filename
        docs_folder = os.path.abspath("./docs")
        file_path = os.path.join(docs_folder, filename)
        try:
            pages = load_document(file_path)
            print("Done Loading")
            docs = split_pages(pages)
            print("Done Splitting")
            add_documents(vector_db, docs)
            
            return jsonify(
                {
                    "message": "File uploaded and processed successfully",
                    "file_id": filename,
                }
            )
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            print("Error processing file:", str(e))
            return jsonify({"error": "Failed to process document"}), 500
    return jsonify({"error": "File upload failed"}), 500


@app.route("/select_embedding", methods=["POST"])
def select_embedding_model():
    global selected_embedding
    data = request.json
    if "embedding_model" not in data:
        return jsonify({"error": "No embedding model specified"}), 400

    selected_embedding = data["embedding_model"]
    return jsonify({"message": f"{selected_embedding} embeddings selected"})


@app.route("/ask", methods=["POST"])
def ask_question():
    global qa_chain, vector_db
    qa_chain = create_chain(vector_db)
    print("Done Creating QA Chain")
    data = request.json
    if not qa_chain:
        return jsonify({"error": "Document not uploaded or processed"}), 400
    if "query" not in data:
        return jsonify({"error": "Invalid input"}), 400

    query = data["query"]
    try:
        answer = qa_chain.run(query)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
