import gradio as gr
import requests
import shutil
import os

API_URL = "http://127.0.0.1:5000"
UPLODAD_FOLDER = "./docs"
history = ""