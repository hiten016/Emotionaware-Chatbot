# Emotion-based-RAG-decision-System

## Objectives
- Detect user intent from natural language queries.
- Identify emotional states to enable context-aware response generation.
- Filter unsafe or harmful inputs using safety classification.
- Improve factual grounding through hybrid retrieval and reranking.
- Reduce hallucinations while maintaining response relevance and quality.
- Build a modular and scalable architecture for real-world conversational AI systems.
  
## Approach

The system consists of four major stages:

- Understanding Layer

User queries are processed by specialized NLP models:

Intent Classification using DeBERTa-v3 fine-tuned on combined dataset(CLINIC, GoEmotion and Toxicity).

These models extract semantic, emotional, and safety-related signals from the query.

- Decision Engine

A policy-driven decision engine combines outputs from the classifiers and determines the appropriate response strategy:

Unsafe queries → Safety refusal.
Knowledge-intensive queries → RAG pipeline.
General conversational queries → Direct LLM generation.
- Hybrid Retrieval Pipeline

For queries requiring external knowledge:

- BM25 retrieves keyword-relevant documents.
- Dense retrieval performs semantic search using vector embeddings stored in Qdrant.
- Hybrid retrieval combines sparse and dense retrieval signals.
-  reranker selects the most relevant documents before generation.
## Response Generation

The retrieved context, intent label, and emotion signals are passed to Qwen3.5-4B for response generation.

