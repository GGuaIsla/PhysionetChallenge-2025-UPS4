# === Base: Python 3.10.1 sobre Debian Buster ===
FROM python:3.10.1-buster

# Evita prompts durante instalación
ENV DEBIAN_FRONTEND=noninteractive

# Instala dependencias del sistema
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1-mesa-glx \
    && apt-get clean

# === Recomendado por TensorFlow para soporte GPU (si no usas base CUDA) ===
# Instalamos TensorFlow con soporte GPU (desde pip, que ya incluye CUDA/cuDNN precompilado)
# Puedes reemplazar tensorflow por tensorflow==2.15.0 por estabilidad
RUN pip install --upgrade pip && \
    pip install tensorflow==2.15.0

# === DO NOT EDIT ===
RUN mkdir /challenge
COPY ./ /challenge
WORKDIR /challenge

# === Instala dependencias del proyecto ===
RUN pip install --no-cache-dir -r requirements.txt

# === Silenciar logs innecesarios ===
ENV TF_CPP_MIN_LOG_LEVEL=2

# === Para asegurar acceso a GPU en ejecución ===
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

