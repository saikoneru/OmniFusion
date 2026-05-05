from typing import List, Dict, Any, Optional
from qwen_omni_utils import process_mm_info


LANG_MAP = {
    "Chinese": "zh", "English": "en", "French": "fr", "German": "de",
    "Italian": "it", "Japanese": "ja", "Korean": "ko", "Portuguese": "pt",
    "Russian": "ru", "Spanish": "es", "Vietnamese": "vi", "Arabic": "ar",
    "Czech": "cs", "Croatian": "hr", "Danish": "da", "Dutch": "nl",
    "Finnish": "fi", "Hungarian": "hu", "Indonesian": "id", "Malay": "ms",
    "Norwegian": "no", "Polish": "pl", "Romanian": "ro", "Turkish": "tr",
}


class MultiModalDataCollator:
    """
    Collator for speech translation with flexible modality combinations.
    
    Supports:
    - Audio-only → translated text
    - Image-only → transcribed/described text → translated text
    - Audio + Image → transcribed audio using image context → translated text

    Each example is expected to have the structure:
        {
            "translation": {
                "audio_fp":   Optional[str],  # path to audio file
                "image_fp":   Optional[str],  # path to image file
                "src":        str,            # source transcription/OCR (optional context)
                "tgt":        str,            # target translation
                "lang":       str,            # target language name, e.g. "German"
            }
        }
    """

    def __init__(self, tokenizer, omni_processor):
        self.tokenizer = tokenizer
        self.omni_processor = omni_processor

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        decoder_inputs, decoder_labels, conversations = [], [], []

        for item in examples:
            ex = item["translation"]
            tgt_lang = ex["lang"]
            lang_tag = LANG_MAP[tgt_lang]

            # ── LLM side (decoder) ──────────────────────────────────────────
            prompt = f"Translate the following sentence into {tgt_lang}:\n <{lang_tag}> "
            decoder_inputs.append(prompt)
            decoder_labels.append(ex["tgt"] + self.tokenizer.eos_token)

            # ── Omni side (encoder) ─────────────────────────────────────────
            # Build conversation based on available modalities
            audio_fp = ex.get("audio_fp")
            image_fp = ex.get("image_fp")

            if audio_fp is not None and image_fp is not None:
                # Audio + Image: Transcribe with visual context
                conversation = [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": (
                            "You are Qwen, a virtual human developed by the Qwen Team, "
                            "Alibaba Group, capable of perceiving auditory and visual inputs, "
                            "as well as generating text and speech."
                        )}],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Transcribe the audio using the image for context and OCR:"},
                            {"type": "image", "image": image_fp},
                            {"type": "audio", "audio": audio_fp},
                        ],
                    },
                ]
            elif audio_fp is not None and image_fp is None:
                # Audio-only: Simple transcription
                conversation = [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": (
                            "You are Qwen, a virtual human developed by the Qwen Team, "
                            "Alibaba Group, capable of perceiving auditory and visual inputs, "
                            "as well as generating text and speech."
                        )}],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Transcribe the audio:"},
                            {"type": "audio", "audio": audio_fp},
                        ],
                    },
                ]
            elif audio_fp is None and image_fp is not None:
                # Image-only: OCR or image description
                conversation = [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": (
                            "You are Qwen, a virtual human developed by the Qwen Team, "
                            "Alibaba Group, capable of perceiving auditory and visual inputs, "
                            "as well as generating text and speech."
                        )}],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Perform OCR on the image or describe the image:"},
                            {"type": "image", "image": image_fp},
                        ],
                    },
                ]
            else:
                raise ValueError(
                    f"Example must have at least audio_fp or image_fp. "
                    f"Got audio_fp={audio_fp}, image_fp={image_fp}"
                )

            conversations.append(conversation)

        # ── Process omni inputs ─────────────────────────────────────────────
        text = self.omni_processor.apply_chat_template(
            conversations, add_generation_prompt=True, tokenize=False
        )
        audios, images, videos = process_mm_info(conversations, use_audio_in_video=False)

        # Resize images if present
        if images is not None:
            images = [
                img.resize((512, 512)) if img is not None else None 
                for img in images
            ]

        omni_inputs = self.omni_processor(
            text=text,
            audio=audios,
            images=images,
            videos=None,
            return_tensors="pt",
            padding="longest",
            truncation=False,
            use_audio_in_video=False,
        )

        # ── Tokenize decoder side ───────────────────────────────────────────
        enc_inputs = self.tokenizer(
            decoder_inputs,
            padding="longest",
            truncation=False,
            return_tensors="pt",
            add_special_tokens=False,
        )
        enc_labels = self.tokenizer(
            decoder_labels,
            padding="longest",
            truncation=False,
            return_tensors="pt",
            add_special_tokens=False,
        )

        return {
            # Omni encoder inputs
            "omni_input_ids":              omni_inputs.input_ids,
            "omni_attention_mask":         omni_inputs.attention_mask,
            "omni_pixel_values":           omni_inputs.get("pixel_values"),
            "omni_image_grid_thw":         omni_inputs.get("image_grid_thw"),
            "omni_input_features":         omni_inputs.get("input_features"),
            "omni_feature_attention_mask": omni_inputs.get("feature_attention_mask"),
            # Decoder inputs / labels
            "text_input_ids":              enc_inputs.input_ids,
            "text_attention_mask":         enc_inputs.attention_mask,
            "labels":                      enc_labels.input_ids,
            "text_labels_attention_mask":  enc_labels.attention_mask,
        }