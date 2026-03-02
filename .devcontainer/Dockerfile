# ──────────────────────────────────────────────────────────────────────────────
# Devcontainer – Passos Mágicos Datathon
# Base: Microsoft Python 3.10 (Debian Bullseye)
# Inclui: TensorFlow, JupyterLab, Streamlit e ferramentas de qualidade de código
# ──────────────────────────────────────────────────────────────────────────────
FROM mcr.microsoft.com/devcontainers/python:3.10-bullseye

# Dependências de sistema necessárias para TensorFlow, HDF5 (keras), matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libhdf5-dev \
        libhdf5-serial-dev \
        pkg-config \
        graphviz \
        git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Portas expostas:
#   8501 → Streamlit
#   8888 → JupyterLab
EXPOSE 8501 8888

# Usuário padrão do devcontainer (não root)
USER vscode

