import streamlit as st
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
import soundfile as sf
from huggingface_hub import login
import os
import tempfile

# Initialize models (same as your original code)
login(token='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')

device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Whisper Model Initialization
whisper_model_id = "openai/whisper-large-v3"
whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
    whisper_model_id, 
    torch_dtype=torch_dtype, 
    low_cpu_mem_usage=True, 
    use_safetensors=True
)
whisper_model.to(device)
whisper_processor = AutoProcessor.from_pretrained(whisper_model_id)

whisper_pipe = pipeline(
    "automatic-speech-recognition",
    model=whisper_model,
    tokenizer=whisper_processor.tokenizer,
    feature_extractor=whisper_processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
    return_timestamps=False
)

# Llama Initialization
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-1B-Instruct",
    task="text-generation",
    max_new_tokens=100,
    do_sample=False,
)
llama = ChatHuggingFace(llm=llm, verbose=True)

# Speech T5 Initialization
synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts")
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

def conversational_pipeline(audio_path, output_path):
    """
    Process audio input through transcription, LLM, and text-to-speech
    
    Args:
        audio_path (str): Path to input audio file
        output_path (str): Directory to save output audio
    
    Returns:
        str: Path to generated audio response
    """
    # Transcribe input audio
    transcription = whisper_pipe(audio_path)["text"]
    st.write(f"🎤 You said: {transcription}")
    
    # Generate LLM response
    messages = [
        ("system", "You are my personal assistant named Jarvis."),
        ("human", transcription)
    ]
    
    llama_response = llama.invoke(messages)
    jarvis_answer = llama_response.content
    st.write(f"🤖 Jarvis responded: {jarvis_answer}")

    # Convert text to speech
    speech = synthesiser(jarvis_answer, forward_params={"speaker_embeddings": speaker_embedding})
    
    # Save audio response
    audio_name = "jarvis_response.wav"
    output_audio_path = os.path.join(output_path, audio_name)
    sf.write(output_audio_path, speech["audio"], samplerate=speech["sampling_rate"])

    return output_audio_path

def main():
    st.title("🎙️ Voice Assistant")
    st.write("Upload an audio file or record a message to talk with Jarvis!")

    # File upload option
    uploaded_file = st.file_uploader("Upload an audio file", type=['wav', 'mp3'])

    # Audio recording option
    record_audio = st.checkbox("Record Audio")
    
    if record_audio:
        # Use Streamlit's built-in audio recorder
        audio_bytes = st.audio_recorder("Click to record", key="recorder")
        
        if audio_bytes:
            # Process the recorded audio
            do_process = st.button("Process Recording")
            if do_process:
                with tempfile.TemporaryDirectory() as tmpdir:
                    # Save the recorded audio
                    input_audio_path = os.path.join(tmpdir, "input_audio.wav")
                    with open(input_audio_path, "wb") as f:
                        f.write(audio_bytes)
                    
                    # Process the audio
                    try:
                        # Generate response audio
                        output_audio_path = conversational_pipeline(
                            audio_path=input_audio_path, 
                            output_path=tmpdir
                        )
                        
                        # Play the response audio
                        with open(output_audio_path, "rb") as audio_file:
                            response_audio_bytes = audio_file.read()
                        
                        st.audio(response_audio_bytes, format="audio/wav")
                    
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
    
    # File upload processing
    if uploaded_file is not None:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save the uploaded audio
            input_audio_path = os.path.join(tmpdir, "input_audio.wav")
            with open(input_audio_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            
            # Process the audio
            try:
                # Generate response audio
                output_audio_path = conversational_pipeline(
                    audio_path=input_audio_path, 
                    output_path=tmpdir
                )
                
                # Play the response audio
                with open(output_audio_path, "rb") as audio_file:
                    response_audio_bytes = audio_file.read()
                
                st.audio(response_audio_bytes, format="audio/wav")
            
            except Exception as e:
                st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()