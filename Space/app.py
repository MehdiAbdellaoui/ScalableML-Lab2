from transformers import pipeline
import gradio as gr

transcription_pipe = pipeline(model="explorall/whisper-small-sv-dropout-6mb")  
translation_pipe = pipeline(model="Helsinki-NLP/opus-mt-sv-en")


def transcribe_and_translate(audio):
    transcription = transcription_pipe(audio)["text"]
    translation = translation_pipe(transcription)[0]['translation_text']
    return transcription, translation

iface = gr.Interface(
    fn=transcribe_and_translate,
    inputs=gr.Audio(source="microphone", type="filepath"),
    outputs=[gr.Textbox(), gr.Textbox()],
    title="Whisper Small Swedish to English Translator",
    description="Realtime demo for Swedish speech recognition using a fine-tuned Whisper small model, and Swedish to English translation using a small T5 model",
)

iface.launch()