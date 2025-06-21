from fastapi import FastAPI, File, UploadFile
from models.emotion_classifier import classify_emotion
from models.stt_transcriber import speech_to_text
from llm.generate_response import generate_reply
from langchain_core.runnables.base import RunnableLambda, RunnableMap
import tempfile
app = FastAPI()

# LangChain Runnables
speech_to_text_chain = RunnableLambda(lambda audio_path: {"text": speech_to_text(audio_path)})
sentiment_chain = RunnableLambda(lambda audio_path: {"sentiment": classify_emotion(audio_path)})

merge_chain = RunnableLambda(lambda inputs: {
    "text": inputs["text"]["text"],
    "sentiment": inputs["sentiment"]["sentiment"]
})

llm_chain = RunnableLambda(generate_reply)

# Full Chain
full_chain = (
    RunnableMap({
        "text": speech_to_text_chain,
        "sentiment": sentiment_chain
    }) | merge_chain | llm_chain
)

@app.post("/process-speech/")
async def process_speech(audio: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp:
        temp.write(await audio.read())
        temp_path = temp.name

    output = full_chain.invoke(temp_path)
    return {
        "response": output
    }