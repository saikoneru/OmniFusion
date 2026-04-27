from typing import List, Dict, Any
from qwen_omni_utils import process_mm_info


LANG_MAP = {
    "Chinese": "zh", "English": "en", "French": "fr", "German": "de",
    "Italian": "it", "Japanese": "ja", "Korean": "ko", "Portuguese": "pt",
    "Russian": "ru", "Spanish": "es", "Vietnamese": "vi", "Arabic": "ar",
    "Czech": "cs", "Croatian": "hr", "Danish": "da", "Dutch": "nl",
    "Finnish": "fi", "Hungarian": "hu", "Indonesian": "id", "Malay": "ms",
    "Norwegian": "no", "Polish": "pl", "Romanian": "ro", "Turkish": "tr",
}


class SpeechTranslationDataCollator:
    """
    Collator for speech translation: audio → translated text.

    Each example is expected to have the structure:
        {
            "translation": {
                "audio_fp": str,          # path to audio file
                "src":      str,          # source transcription (optional context)
                "tgt":      str,          # target translation
                "lang":     str,          # target language name, e.g. "German"
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
            conversations.append([
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
                        {"type": "audio", "audio": ex["audio_fp"]},
                    ],
                },
            ])

        # ── Process omni inputs ─────────────────────────────────────────────
        text = self.omni_processor.apply_chat_template(
            conversations, add_generation_prompt=True, tokenize=False
        )
        audios, images, videos = process_mm_info(conversations, use_audio_in_video=False)

        omni_inputs = self.omni_processor(
            text=text,
            audio=audios,
            images=None,
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
            "omni_input_ids":            omni_inputs.input_ids,
            "omni_attention_mask":       omni_inputs.attention_mask,
            "omni_input_features":       omni_inputs.get("input_features"),
            "omni_feature_attention_mask": omni_inputs.get("feature_attention_mask"),
            # Decoder inputs / labels
            "text_input_ids":            enc_inputs.input_ids,
            "text_attention_mask":       enc_inputs.attention_mask,
            "labels":                    enc_labels.input_ids,
            "text_labels_attention_mask": enc_labels.attention_mask,
        }
