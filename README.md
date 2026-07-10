# RAG Chatbot

A retrieval-augmented generation chatbot. Upload documents (PDF, TXT, or Markdown), ask
questions, and get answers from your content with the source passages cited. It only answers
from the documents, so it does not make things up.

## What it does
- Reads PDF, TXT, and Markdown files and splits them into overlapping passages.
- Embeds them locally with all-MiniLM-L6-v2, so the documents stay on the machine.
- Retrieves the most relevant passages for each question by cosine similarity.
- Answers with Groq or OpenAI, using only the retrieved passages, with inline [1] [2] citations.
- Shows the source passages behind every answer.
- Runs without an API key in retrieval-only mode, so you can see the grounding for free.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
In the sidebar, pick a provider and paste a key:
- Groq: free key at https://console.groq.com
- OpenAI: an OPENAI_API_KEY
- Or leave it on Retrieval only to run without a key.

Upload your own documents, or tick "Use the sample documents" to try it on the included
Acme Robotics handbook (for example: "What is the warranty period?" or "Can I return a robot?").

## Deploy a live demo (free)
1. Push the repo to GitHub.
2. Open share.streamlit.io, create a new app, and point it at app.py.
3. Optionally add GROQ_API_KEY in the app secrets so visitors get generated answers.

## Configuration
Set keys in the sidebar, as environment variables (GROQ_API_KEY / OPENAI_API_KEY), or in
.streamlit/secrets.toml (see .streamlit/secrets.toml.example).

## Stack
Python, Streamlit, sentence-transformers, NumPy, Groq/OpenAI

Built by Satyawan Singh.
