import torch
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from PIL import Image

from transformers import PreTrainedTokenizer, PreTrainedModel
from qwen_omni_utils import process_mm_info
from transformers import Qwen2_5OmniProcessor   # modify import if needed


@dataclass
class TranslationInput:
    audio_path: Optional[str] = None
    image_path: Optional[str] = None
    source_text: str = ""


LANG_MAP = {
    "Chinese": "zh", "English": "en", "French": "fr", "German": "de",
    "Italian": "it", "Japanese": "ja", "Korean": "ko", "Portuguese": "pt",
    "Russian": "ru", "Spanish": "es", "Vietnamese": "vi", "Arabic": "ar",
    "Czech": "cs", "Croatian": "hr", "Danish": "da", "Dutch": "nl",
    "Finnish": "fi", "Hungarian": "hu", "Indonesian": "id", "Malay": "ms",
    "Norwegian Bokmal": "nb", "Norwegian": "no", "Polish": "pl",
    "Romanian": "ro", "Turkish": "tr"
}


class OmniFusionProcessor:
    """
    Processor that merges Omni model multimodal inputs with
    a text model (Seed / Mistral / etc.) side-channel input.
    """

    SYSTEM_PROMPT = (
        "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
        "capable of perceiving auditory and visual inputs, as well as generating "
        "text and speech."
    )
    IMAGE_SIZE = (512, 512)
    USE_AUDIO_IN_VIDEO = True

    def __init__(
        self,
        omni_processor: Qwen2_5OmniProcessor,
        text_tokenizer: PreTrainedTokenizer,
        device: torch.device
    ):
        self.omni_processor = omni_processor
        self.text_tokenizer = text_tokenizer
        self.device = device

    # -----------------------------------------------------------------------
    # Conversation Builder
    # -----------------------------------------------------------------------
    def _build_conversation(
        self,
        translation_input: TranslationInput,
        target_lang: str,
        use_cot: bool = False
    ) -> Tuple[List[Dict], str]:

        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.SYSTEM_PROMPT}],
            }
        ]

        lang_tag = LANG_MAP[target_lang]
        prefix_tower = ""
        suffix_tower = f" <{lang_tag}> "

        # ----------------------------
        # AUDIO CASE
        # ----------------------------
        if translation_input.audio_path is not None:
            content = [
                {
                    "type": "text",
                    "text": "Transcribe the audio using the image for context along with OCR on it for spelling keywords and names: "
                }
            ]

            if translation_input.image_path:
                content.append({"type": "image", "image": translation_input.image_path})

            content.append({"type": "audio", "audio": translation_input.audio_path})

            conversation.append({"role": "user", "content": content})

            if use_cot and translation_input.source_text:
                suffix_tower = translation_input.source_text + suffix_tower

        # ----------------------------
        # IMAGE + OPTIONAL TEXT
        # ----------------------------
        elif translation_input.image_path is not None:
            if translation_input.source_text == "":
                suffix_tower = " <OCR> "
            else:
                prefix_tower = (
                    f" <SRC> {translation_input.source_text}{self.text_tokenizer.eos_token}"
                )

            conversation.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"Translate the following sentence into {target_lang} "
                                "using the image for context. <IMG> denotes the image. "
                                "Use the image for formality, gender, keywords, and "
                                "important disambiguation cues.\n"
                                f"{translation_input.source_text} <IMG>"
                            ),
                        },
                        {"type": "image", "image": translation_input.image_path},
                    ],
                }
            )

        text_input = prefix_tower + suffix_tower
        return conversation, text_input

    # -----------------------------------------------------------------------
    # Image Resize
    # -----------------------------------------------------------------------
    def _resize_images(self, images: Optional[List]) -> Optional[List]:
        if images is None:
            return None
        return [img.resize(self.IMAGE_SIZE) if img is not None else None for img in images]

    # -----------------------------------------------------------------------
    # Batch Processor
    # -----------------------------------------------------------------------
    def prepare_batch(
        self,
        inputs: List[TranslationInput],
        target_lang: str,
        use_cot: bool = False,
    ) -> Dict[str, torch.Tensor]:

        conversations = []
        text_inputs = []

        # Build conversations
        for ti in inputs:
            conv, txt = self._build_conversation(ti, target_lang, use_cot)
            conversations.append(conv)
            text_inputs.append(txt)

        # Omni's multimodal encoding
        text = self.omni_processor.apply_chat_template(
            conversations, add_generation_prompt=False, tokenize=False
        )

        audios, images, videos = process_mm_info(
            conversations, use_audio_in_video=self.USE_AUDIO_IN_VIDEO
        )

        images = self._resize_images(images)

        omni_inputs = self.omni_processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding="longest",
            truncation=False,
            use_audio_in_video=self.USE_AUDIO_IN_VIDEO,
        ).to(self.device)

        # Tokenize the text tower inputs
        text_inputs_tok = self.text_tokenizer(
            text_inputs,
            max_length=512,
            return_tensors="pt",
            add_special_tokens=False,
            padding="longest",
        ).to(self.device)

        # Combine token IDs
        concat_ids = self._concatenate_input_ids(omni_inputs, text_inputs_tok)

        return self._build_final_inputs(omni_inputs, text_inputs_tok, concat_ids)

    # -----------------------------------------------------------------------
    # ID Concatenation
    # -----------------------------------------------------------------------
    def _concatenate_input_ids(self, omni_inputs, text_inputs):
        omni_nopad = [
            seq[seq != self.omni_processor.tokenizer.pad_token_id]
            for seq in omni_inputs.input_ids
        ]
        text_nopad = [
            seq[seq != self.text_tokenizer.pad_token_id]
            for seq in text_inputs.input_ids
        ]

        merged = [torch.cat([o, t], dim=0) for o, t in zip(omni_nopad, text_nopad)]
        max_len = max(seq.size(0) for seq in merged)

        padded = torch.full(
            (len(merged), max_len),
            self.text_tokenizer.pad_token_id,
            dtype=merged[0].dtype,
            device=merged[0].device,
        )

        for i, seq in enumerate(merged):
            padded[i, -seq.size(0):] = seq

        return padded

    # -----------------------------------------------------------------------
    # Output Dict Builder
    # -----------------------------------------------------------------------
    def _build_final_inputs(self, omni_inputs, text_inputs, concat_ids):
        return {
            "omni_input_ids": omni_inputs.input_ids,
            "omni_attention_mask": omni_inputs.attention_mask,
            "omni_pixel_values": (
                [omni_inputs.pixel_values] if "pixel_values" in omni_inputs else None
            ),
            "omni_image_grid_thw": (
                [omni_inputs.image_grid_thw] if "image_grid_thw" in omni_inputs else None
            ),
            "omni_input_features": (
                omni_inputs.input_features if "input_features" in omni_inputs else None
            ),
            "omni_feature_attention_mask": (
                omni_inputs.feature_attention_mask
                if "feature_attention_mask" in omni_inputs
                else None
            ),
            "text_input_ids": text_inputs.input_ids,
            "text_attention_mask": text_inputs.attention_mask,
            "input_ids": concat_ids,
            "attention_mask": (concat_ids != self.text_tokenizer.pad_token_id).long(),
        }

