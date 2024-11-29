import gradio as gr
import requests
import shutil
import os

API_URL = "http://127.0.0.1:5000"
UPLODAD_FOLDER = "./docs"
history = ""

def upload_document(file):
    try:
        if not os.path.exists(UPLODAD_FOLDER):
            os.makedirs(UPLODAD_FOLDER)
        shutil.copy(file, UPLODAD_FOLDER)
        files = {'file': file}
        gr.Info("Wait while uploading the document and processing ...")
        response = requests.post(f"{API_URL}/upload", files=files)
        if response.status_code != 200:
            return f"Error: {response.text}"
        gr.Info(f"Done Uploading and Preprocessing ...")
        return gr.update(interactive=True)
    except requests.exceptions.RequestException as e:
        return f"Request failed: {str(e)}"
    except ValueError as e:
        return f"Failed to parse response: {str(e)}"