# Emotionaware-API-for-LLMs

Emotionaware-API-for-LLMs is a production-style API layer for LLMs that enhances responses using emotion awareness, intent detection, safety filtering, and a policy-driven decision engine. It dynamically routes user queries based on emotional and contextual signals to generate empathetic, safe, and contextually appropriate responses without modifying the underlying model.

The system uses a modular architecture consisting of an understanding layer (intent, emotion, safety models), a core decision policy engine that determines response strategy, and an action router that executes LLM calls, safety refusals, or optional retrieval-augmented generation (RAG). A feedback loop collects user ratings and interaction signals to improve future decision-making.

## System Architecture

```mermaid
flowchart LR

A[User Multi-turn Input] --> B[Understanding Layer]

B --> C1[Intent Detection<br/>Question, Complaint, Request]
B --> C2[Emotion Detection<br/>Anger, Sadness, Neutral, Joy]
B --> C3[Safety Classifier<br/>Safe / Unsafe]

C1 --> D
C2 --> D
C3 --> D

D[Decision Policy]

D --> E1[Standard Response Path]
D --> E2[Empathetic Response Path]
D --> E3[Safety Response / Refusal Path]

E1 --> F[Prompt Conditioning Layer]
E2 --> F
E3 --> F

F --> G[LLM Generator<br/>Qwen 3.5 4B]

G --> H[Final Response]
```
