# OmniFusion 

## 📄 Publication

### OmniFusion: Simultaneous Multilingual Multimodal Translations via Modular Fusion  
**Sai Koneru, Matthias Huck, Jan Niehues**

[![arXiv](https://img.shields.io/badge/arXiv-2512.00234-b31b1b.svg)](https://arxiv.org/abs/2512.00234)
[![HuggingFace Paper](https://img.shields.io/badge/HF%20Paper-2512.00234-ffcc00.svg)](https://huggingface.co/papers/2512.00234)
[![GitHub](https://img.shields.io/badge/GitHub-OmniFusion-181717.svg)](https://github.com/saikoneru/OmniFusion)
[![HuggingFace Model](https://img.shields.io/badge/HF%20Model-OmniFusion-ff6f00.svg)](https://huggingface.co/skoneru/OmniFusion)

Guide on how to run **OmniFusion** for image, audio, and text inference with code or as gradio demo.

---

## 🎥 Demo: Integration in KIT Lecture Translator

This demo illustrates how **OmniFusion** can be integrated for:

- **Simultaneous Multimodal Speech Translation**  
  (speech → text translation during live lectures)

- **Slide Translation**  
  using the *Image Translator*  for image → image translation in slides


<div style="width: 800px; height: 400px; margin: 0 auto;">
  <video autoplay muted loop controls playsinline style="width: 100%; height: 100%; object-fit: contain;">
    <source src="https://huggingface.co/skoneru/OmniFusion/resolve/main/boom_lt.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>
</div>

Related systems and references:
- Image Translator (for slide translation): https://github.com/saikoneru/image-translator  
- KIT Lecture Translator middleware: https://gitlab.kit.edu/kit/isl-ai4lt/lt-middleware/ltpipeline  
- LT system paper (EMNLP 2023 Demo): https://aclanthology.org/2023.emnlp-demo.2.pdf
- BOOM: Beyond Only One Modality KIT's Multilingual Multimodal Lecture Companion (in review)


## Note

What the model is trained for:

1. Relatively Clean/ Single Speaker Speech or Speech + Image Translation
2. Caption Translation (Text describing the image)

What the model is **not** trained for:

1. Multi Speaker Noisy Audio Speech Translation
2. Text in Image Translation (we observe tendency to do OCR or translate whole text when given partial input)

## 1. Installation

### Step 1 — Clone the Repository

```bash
git clone https://github.com/saikoneru/OmniFusion
cd OmniFusion
```

### Step 2. — Install Dependencies

Install all dependencies required by Qwen2.5-Omni-7B (listed on):

https://huggingface.co/Qwen/Qwen2.5-Omni-7B

Use transformers 4.56.1 (this version has been tested).

Additionally install gradio

```bash
pip install gradio
```

## 2. Sample Inference

### Load Model

Run the following python code in the root directory of the repo.

```python
from omni_fusion_model import OmniFusionModel

system = OmniFusionModel()

```

### Audio-Only Translation

```python
audio_path = "examples/speech.wav"

translations = system.translate(
    audio_paths=[audio_path],
    image_paths=[],        
    source_texts=[""],     
    target_lang="English",
)

print("Transcription:", translations[0])
```

### Audio-Image Translation

```python
translations = system.translate(
    audio_paths=["examples/doctor.wav"],
    image_paths=["examples/prescription.jpg"],
    source_texts=[""],      
    target_lang="English",
)

print("Output:", translations[0])
```

### Text-Image Translation

```python
image_path = "examples/label.png"

translations = system.translate(
    audio_paths=[],
    image_paths=[image_path],
    source_texts=["this is a example"],          
    target_lang="English",
)

print("Translation:", translations[0])
```

> **Note:**  
> `source_text` should only be used when **no audio** is provided.  
> If `audio_paths` contains any audio input, then `source_text` **must be an empty string** (`""`) as it will be ignored.
