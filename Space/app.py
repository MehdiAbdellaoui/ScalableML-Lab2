from transformers import pipeline
import gradio as gr
from pytube import YouTube

transcription_pipe = pipeline(model="explorall/whisper-small-sv-dropout-6mb")  
translation_pipe = pipeline(model="Helsinki-NLP/opus-mt-sv-en")

def transcribe_and_translate(audio):
    transcription = transcription_pipe(audio)["text"]
    translation = translation_pipe(transcription)[0]['translation_text']
    
    return transcription, translation

def transcribe_and_translate_yt(link):
    yt = YouTube(link)
    audio = yt.streams.filter(only_audio=True).first().download()

    return transcribe_and_translate(audio)
    
with gr.Blocks() as demo:
    with gr.Tab("Real-time Swedish to English Transcription and Translation"):
        audio = gr.Audio(source="microphone", type="filepath")
        rt_outputs = [gr.Textbox(), gr.Textbox()]
        rt_button = gr.Button('Transcribe and Translate')

    with gr.Tab("Youtube Video Transcription and Translation"):
        link = gr.Textbox(label="Enter YouTube Video Link")
        yt_outputs = [gr.Textbox(), gr.Textbox()]
        yt_button = gr.Button('Transcribe and Translate YouTube Video')

    rt_button.click(transcribe_and_translate, inputs=audio, outputs=rt_outputs)
    yt_button.click(transcribe_and_translate_yt, inputs=link, outputs=yt_outputs)
        
demo.launch(debug=True)