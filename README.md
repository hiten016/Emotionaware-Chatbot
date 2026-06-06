# Emotionaware-API-for-LLMs

This project builds an API layer for LLMs that improves responses by adding emotion awareness, intent detection, safety handling, and user feedback learning. It generates more empathetic, safe, and context-aware responses without modifying the base model. The system first analyzes user input to detect emotion and intent (question, rant, help-seeking). Based on this, it dynamically routes and conditions prompts to guide the LLM toward appropriate responses (empathetic tone, safe refusal). A built-in feedback loop collects user ratings and converts them into training data to improve the model. 
## System Architecture

```mermaid
flowchart LR

A[User Multi-turn Input] --> B[Safety + Intent Classifier<br/>DeBERTa-v3 (CLINC150)<br/>Intent Accuracy: 91.7%<br/>Safety F1: 93.1%]

B --> C1[Safe / Allowed Request Routing]
B --> C2[Unsafe / Risky Detection]

C1 --> D[Context Builder<br/>Conversation Memory<br/>Optional RAG / Tools]

C2 --> E[Safety Response / Refusal Layer]

D --> F[Qwen3.5 4B Generator<br/>Context-aware response generation<br/>Tool-augmented reasoning (optional)]

F --> G[Output + Logging + Feedback Capture]

G --> H[Feedback Pipeline<br/>5K+ interactions<br/>Error tagging<br/>Safety regression tracking<br/>Continuous refinement signals]
```
