from flask import Flask, request, render_template, send_file, session
import soundfile as sf
import os
import torch
from transformers import pipeline, AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from huggingface_hub import login

# Login to Hugging Face
login(token='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')

dashline = '---' * 20

app = Flask(__name__)

# Set a secret key for sessions
app.secret_key = os.urandom(24)

# Device and precision settings
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Whisper Initialization
whisper_model_id = "openai/whisper-small"
whisper_pipe = pipeline(
    "automatic-speech-recognition",
    model=whisper_model_id,
    tokenizer=whisper_model_id,
    device=device,
    torch_dtype=torch_dtype,
)

# Llama Initialization
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-1B-Instruct",
    task="text-generation",
    max_new_tokens=1000,
    do_sample=False,
)
llama = ChatHuggingFace(llm=llm, verbose=True)

# Parler Initialization
parler = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-v1.1").to(device)
tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1.1")
description_tokenizer = AutoTokenizer.from_pretrained(parler.config.text_encoder._name_or_path)


description = (
    "A male speaker delivers a slightly expressive and animated speech with a "
    "moderate speed and pitch. The recording is of very high quality, with the "
    "speaker's voice sounding clear and very close up."
)

# Default conversation memory in session
@app.before_request
def initialize_conversation():
    if 'conversation_history' not in session:
        session['conversation_history'] = [
            ("system", "You are my personal assistant named Jarvis. You are an expert in Machine Learning and Computer Science.")
        ]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/process_audio", methods=["POST"])
def process_audio():

    conversation_history = session['conversation_history']


    audio_file = request.files["audio"]
    audio_path = "uploaded_audio.wav"
    audio_file.save(audio_path)


    print(dashline)
    print("Processing audio with Whisper...")
    transcription = whisper_pipe(audio_path)["text"]
    print(f"User said: {transcription}")
    print(dashline)


    conversation_history.append(("human", transcription))


    llama_response = llama.invoke(conversation_history)
    jarvis_answer = llama_response.content
    print(f"Jarvis response: {jarvis_answer}")
    conversation_history.append(("assistant", jarvis_answer))
    print(dashline)


    session['conversation_history'] = conversation_history


    print("Generating audio reply...")
    input_ids = description_tokenizer(description, return_tensors="pt").input_ids.to(device)
    prompt_input_ids = tokenizer(jarvis_answer, return_tensors="pt").input_ids.to(device)
    generation = parler.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
    audio_arr = generation.cpu().numpy().squeeze()


    response_audio_file = "static/response_audio.wav"
    sf.write(response_audio_file, audio_arr, parler.config.sampling_rate)
    print(f"Audio reply saved to {response_audio_file}")
    print(dashline)


    return {"audio_url": "/static/response_audio.wav"}


@app.route("/process_chat", methods=["POST"])
def process_chat():

    conversation_history = session['conversation_history']


    user_message = request.json.get("message")
    print(f"User message: {user_message}")
    
    if not user_message:
        return {"error": "No message provided"}, 400


    conversation_history.append(("human", user_message))


    print("Conversation history before LLM response:")
    print(conversation_history)

    try:
        llama_response = llama.invoke(conversation_history)
        jarvis_answer = llama_response.content
        print(f"Jarvis response: {jarvis_answer}")
    except Exception as e:
        print(f"Error during Llama response generation: {e}")
        return {"error": "Failed to generate response from the assistant"}, 500

    conversation_history.append(("assistant", jarvis_answer))

    session['conversation_history'] = conversation_history

    return {"assistant_text": jarvis_answer}


if __name__ == "__main__":
    app.run(debug=True)
