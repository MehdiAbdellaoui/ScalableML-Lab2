from transformers import pipeline
import gradio as gr

transcribe = pipeline(model="Scalable-ML/whisper-small-sv")
translate = pipeline(model="Scalable-ML/t5-small")

def transcribe_and_translate(audio):
    text = transcribe(audio)["text"]
    translation = translate(text)
    return text, translation

iface = gr.Interface(
    fn=transcribe_and_translate,
    inputs=gr.Audio(source="microphone", type="filepath"),
    outputs="text",
    title="Whisper Small Swedish",
    description="Realtime demo for Swedish speech recognition using a fine-tuned Whisper small model.",
)

iface.launch()