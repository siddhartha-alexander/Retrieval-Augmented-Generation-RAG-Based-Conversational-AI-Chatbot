# Retrieval-Augmented Generation (RAG) Conversational AI Chatbot
# Overview

This project implements a Retrieval-Augmented Generation (RAG) based conversational AI system that answers user queries by retrieving relevant information from external documents and generating grounded responses using a Large Language Model (LLM).

# Objective

Traditional LLM chatbots often hallucinate and perform poorly on document-specific queries. This project solves that by grounding responses in retrieved document context.

# System Architecture
PDF Documents
   ↓
Text Extraction
   ↓
Text Chunking
   ↓
Embedding Generation
   ↓
Chroma Vector Database
   ↓
Similarity Search
   ↓
Prompt Augmentation
   ↓
LLM Response

# Tech Stack

Python

LangChain

Chroma Vector Database

HuggingFace Transformers

PyPDFLoader

Gradio

# Data Pipeline
# Document Ingestion
PDF files are loaded and processed for downstream embedding.

# Text Chunking
Documents are split into overlapping chunks to preserve context.

# Vector Storage
Embeddings are stored in Chroma for similarity search.

# Retrieval & Generation
User queries are embedded and matched against document vectors. Retrieved context is injected into the LLM prompt to generate grounded answers.

# Prompt Design
The model is instructed to answer only from retrieved context, reducing hallucinations.

# Conversational Interface
The chatbot includes an interactive Gradio-based chat UI with conversation history support.

# Evaluation
Manual testing shows improved factual accuracy and relevance compared to vanilla LLM responses.

# Results

Reduced hallucinations
Improved relevance
Accurate document-grounded answers

Accurate document-grounded answers
