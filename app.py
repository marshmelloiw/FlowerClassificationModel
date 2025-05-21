from fastai.vision.all import *
import gradio as gr

learn = load_learner("flower_classifier.pkl")

def classify_flower(img):
    pred, idx, probs = learn.predict(img)
    return {learn.dls.vocab[i]: float(probs[i]) for i in range(len(probs))}

css = """
body {
    background: #bb295a;
    font-family: 'Lucida Console', monospace;
}
h1 {
    color: #ffb4cd;
    font-size: 2rem;
}
.output_label {
    color: #b51b75;
}
.gradio-container {
    border: 2px dashed #ffb7d5;
    border-radius: 20px;
    padding: 20px;
    background: #6a1f38;
}
footer {
    visibility: hidden;
}
.gr-interface .prose > p {
    color: #4b004b !important;   /* Daha koyu mor */
    font-size: 1.6rem !important;
    font-weight: bold !important;
    font-family: 'Lucida Console', monospace !important;
}
"""

interface = gr.Interface(
    fn=classify_flower,
    inputs=gr.Image(type="pil", label="🌸 Upload a flower image 🌸"),
    outputs=gr.Label(num_top_classes=3, label="🌼 Prediction 🌼"),
    title="🌸 Flower Classifier 🌸",
    description="🌷 Upload a flower image and discover which of the 17 Oxford flower classes it belongs to! 🌻",
    css=css
)

interface.launch()
