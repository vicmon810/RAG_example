#!/bin/bash 

<<<<<<< HEAD
python app.py

=======
set -e #error warning

echo "== Starting RAG system ==" 

if [! d "llm"]; then
    echo "Creating virtual environment."
    pyhton3 -m venv llm 
fi 

source llm/bin/activate 

pip install --upgrade pip

if [-f "requirements.txt"]; then
    echo "Installing dependencies " 
    pip install -r requirements.txt 
else
    echo "requirement.txt not found"
    exit 1
fi 

if ! curl -s http://localhost:11433 > /dev/null; then   
    echo "Ollama not running. starting ollama "
    ollama serve &
    sleep 3
fi 

echo "Loading model "
ollama run qwen3.5:0.8 "" > /dev/null 2>&1 || true 

echo "Running "
python app.py
>>>>>>> 2d011645342db4e844cd6c6a421fb2eef2458268
