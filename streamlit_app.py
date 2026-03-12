"""
Entry point para o Streamlit Community Cloud.
O Streamlit Cloud detecta automaticamente este arquivo na raiz do repositório.
O app real está em app/app.py.
"""
import os
import runpy
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.join(ROOT_DIR, "app")
APP_PATH = os.path.join(APP_DIR, "app.py")

if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

runpy.run_path(APP_PATH, run_name="__main__")

