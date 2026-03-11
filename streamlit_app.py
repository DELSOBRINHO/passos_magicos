"""
Entry point para o Streamlit Community Cloud.
O Streamlit Cloud detecta automaticamente este arquivo na raiz do repositório.
O app real está em app/app.py.
"""
import runpy
import os

app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app", "app.py")
runpy.run_path(app_path)

