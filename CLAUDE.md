# Smile AI - Project Instructions

## Overview
Smile is an AI-powered virtual assistant that performs text-based tasks including document analysis, keyword extraction, Q&A, and summarization.

## Project Structure
- `index.html` — Static landing page / website
- `smile.py` — Main Streamlit application
- `mini_robot.png` — Robot mascot avatar
- `requirements.txt` — Python dependencies

## Running the Streamlit App
```bash
pip install -r requirements.txt
streamlit run smile.py
```

## Opening the Website
Open `index.html` directly in a browser — no build step required.

## Key Technologies
- **Streamlit** for the interactive app UI
- **KeyBERT** for keyword extraction
- **Hugging Face Transformers** (BART for summarization, RoBERTa for Q&A)
- **OpenAI API** for general-knowledge chat
- **LangChain / Pinecone** for advanced LLM orchestration
