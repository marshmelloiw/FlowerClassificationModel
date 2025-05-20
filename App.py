import gradio as gr
from fastai.vision.all import *

learn = load_learner("flower_classifier.pkl")

def classify_image(img):
    pred, idx, probs = learn.predict(img)
    return {learn.dls.vocab[i]: float(probs[i]) for i in range(len(probs))}

interface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="🌸 Flower Classification App 🌸",
    description="Upload a flower image.",
    examples=None
)

interface.launch()