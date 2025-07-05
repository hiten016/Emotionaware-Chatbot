from model_load import predict_emotions
from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")

def gpt_response(user_input, emotions):
    emotion_text = ', '.join(emotions)
    prompt = f"The user is feeling {emotion_text}. Respond empathetically.\nUser: {user_input}\nBot:"
    output = generator(prompt, max_new_tokens=50, do_sample=True, temperature=0.9)[0]['generated_text']
    return output.split("Bot:")[-1].strip()

print("Emotion-Aware Chatbot (type 'quit' to exit)")
while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break
    emotions = predict_emotions(user_input)
    print(f"[Detected Emotions: {emotions}]")
    response = gpt_response(user_input, emotions)
    print("Bot:", response)
