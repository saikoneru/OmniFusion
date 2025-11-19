# OmniFusion 

Guide on how to run **OmniFusionModel** for image, audio, and text inference with code or as gradio demo.

---

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

