from transformers import pipeline

chat_model = pipeline("text-generation", model="gpt2")  # Replace with LLM like Mistral, LLaMA, etc.

def generate_reply(emotion: str, transcript: str) -> str:
    prompt = f"The user sounds {emotion} and said: '{transcript}'. Respond appropriately."
    output = chat_model(prompt, max_length=100, do_sample=True)[0]['generated_text']
    return output[len(prompt):].strip()  # Remove prompt from output