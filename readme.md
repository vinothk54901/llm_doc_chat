# 🧠 RAG-Based Document Chatbot

This repository contains a Retrieval-Augmented Generation (RAG) based chatbot application. It enables intelligent question-answering and summarization over diverse data sources including documents, directories, and websites. The application leverages vector-based retrieval for efficient and accurate responses, and features a user-friendly Streamlit frontend with voice chat support.

## 🏗️ Architecture Diagram
![Architecture Diagram](./Images/arc.drawio.svg) 

## 🚀 Features


1. **📄 Document Summarization**
   - Upload individual files (PDF) to generate concise summaries using LLM-powered summarization.

2. **📁 Directory-Based Q&A**
   - Point to a directory containing documents. The app recursively indexes all supported files and makes them available for semantic search and question-answering.

3. **🌐 Website Chat**
   - Provide a URL to extract and index web content. Ask questions about the site’s content in natural language.

4. **📝 Text Summarization**
   - Paste or input raw text to receive a clean, concise summary.

5. **🎙️ Voice Chat**
   - Ask questions using voice input and receive spoken responses using text-to-speech.

## 🧰 Tech Stack

| Component       | Technology                        |
|------------------|------------------------------------|
| Backend          | Python, LangChain, FAISS |
| Frontend         | Streamlit                         |
| LLM Interface    | OpenAI / HuggingFace Transformers |
| Data Sources     | Local files, Directories, Web URLs |
| Voice Support    | gTTS |
| LLM Abstraction  | Factory Pattern for LLM Providers |

## 🏗️ LLM Provider Abstraction

To support multiple LLM providers seamlessly, the application uses the **Factory Pattern**. This design allows dynamic selection of the language model backend (e.g., OpenAI or Hugging Face) based on selection from Streamlit frontend.

## To DO:
Support multiple document types like excel,ppt,docx and support complex tables , images inside a document

## 🏁 Getting Started

Follow these steps to set up the project locally:

### 1. Create a Virtual Environment

```bash
python -m venv venv
```

### 2. Activate the Virtual Environment

- On **Windows**:
  ```bash
  venv\Scripts\activate
  ```

- On **macOS/Linux**:
  ```bash
  source venv/bin/activate
  ```

### 3. Clone the Repository

```bash
git clone https://github.com/vinothk54901/llm_doc_chat.git
cd llm_doc_chat
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure Environment Variables

Create a `.env` file in the root directory and add your API keys:

```env
OPENAI_API_KEY=your_openai_key
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token
GITHUB_PERSONAL_ACCESS_TOKEN=your_github_token
```

### 6. Run the Application

```bash
streamlit run app.py
```
