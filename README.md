skyniche-assignment


This project is a Streamlit-based application that allows users to interact with documents using LLMs and embedding models.

1. Environment Setup

First, create and activate a virtual environment to keep your dependencies isolated:

```bash
Create the environment
python -m venv venv

Activate the environment
On Windows:
venv\Scripts\activate
On macOS/Linux:
source venv/bin/activate

```

2. Installation

Install the necessary Python packages using pip:

```bash
pip install -r requirements.txt

```

3. Configuration

Create a file named `.env` in the root directory and add the following variables. Replace the placeholders with your specific model names and paths:

```env
Directory where your documents are stored
DOCUMENT_DIR=

Hugging Face LLM Model ID (e.g., mistralai/Mistral-7B-v0.1)
LLM_MODEL='your-model-id-here'

Embedding Model Name (e.g., sentence-transformers/all-MiniLM-L6-v2)
EMBEDDING_MODEL='your-embedding-model-name'

```

> Note: Ensure you create a folder in your project root and place your source files inside it.

4. Running the Application

Launch the Streamlit interface directly from your terminal:

```bash
streamlit run app.py

```
