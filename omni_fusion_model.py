import time
import torch
from typing import List, Optional, Tuple, Dict

from transformers import (
    AutoTokenizer,
    Qwen2_5OmniProcessor,
    Qwen2_5OmniThinkerForConditionalGeneration,
)
from peft import PeftModel

from models.fusable_omni_mistral import FusableMistralForCausalLM
from omni_fusion_processor import OmniFusionProcessor, TranslationInput, LANG_MAP


# -------------------------------------------------------
# Utility: Resize tokenizer + embeddings when adding tokens
# -------------------------------------------------------
def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer,
    model,
):
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_avg
        output_embeddings[-num_new_tokens:] = output_avg


# -------------------------------------------------------
# Standalone OmniFusionModel
# -------------------------------------------------------
class OmniFusionModel:
    """
    Standalone multimodal translation + transcription system
    combining Qwen2.5 Omni with a Fusable Mistral text model.
    """

    def __init__(
        self,
        cache_dir: str = "",
        checkpoint_path: str = "skoneru/OmniFusion",
        device: str = "cuda:0",
    ):
        self.cache_dir = cache_dir
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device)

        print("Loading models...")
        self._load_models()
        self._setup_processor()
        print("Models loaded successfully!")

    # ---------------------------------------------------
    # Load Omni + Fusion + Tokenizers
    # ---------------------------------------------------
    def _load_models(self):
        # 1. Load Qwen2.5 Omni model (vision + audio + text)
        self.omni_model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-Omni-7B",
            dtype=torch.bfloat16,
            trust_remote_code=True,
            cache_dir=self.cache_dir,
            device_map=str(self.device),
        )

        # 2. Omni processor (handles chat template, multimodal encoding)
        self.omni_processor = Qwen2_5OmniProcessor.from_pretrained(
            "Qwen/Qwen2.5-Omni-7B",
            trust_remote_code=True,
            cache_dir=self.cache_dir,
        )
        self.omni_processor.tokenizer.padding_side = "left"

        # 3. Text tokenizer (Seed-X PPO)
        self.tokenizer = AutoTokenizer.from_pretrained(
            "ByteDance-Seed/Seed-X-PPO-7B",
            cache_dir=self.cache_dir,
        )
        self.tokenizer.padding_side = "left"

        # 4. Fused Mistral model (Seed text + Omni multimodal fusion)
        self.fuse_model = FusableMistralForCausalLM.from_pretrained(
            "ByteDance-Seed/Seed-X-PPO-7B",
            cache_dir=self.cache_dir,
            device_map=str(self.device),
            torch_dtype=torch.bfloat16,
            omni_model=self.omni_model,
            omni_embed_dim=3584,
            depth=2,
            mode="gated",
        )

        # 5. Add pad token if tokeniser lacks one
        if self.tokenizer.pad_token is None:
            print("Tokenizer has no pad token, inserting `<pad>`")
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict={"pad_token": "<pad>"},
                tokenizer=self.tokenizer,
                model=self.fuse_model,
            )

        # 6. Load LoRA/PEFT weights and merge
        self.fuse_model = PeftModel.from_pretrained(
            self.fuse_model,
            self.checkpoint_path,
        )

        self.fuse_model = self.fuse_model.merge_and_unload()

        # Freeze everything
        for p in self.fuse_model.parameters():
            p.requires_grad = False

    # ---------------------------------------------------
    # Setup OmniFusionProcessor (input builder)
    # ---------------------------------------------------
    def _setup_processor(self):
        self.processor = OmniFusionProcessor(
            omni_processor=self.omni_processor,
            text_tokenizer=self.tokenizer,
            device=self.device,
        )

    # ---------------------------------------------------
    # Public API: Translate batch
    # ---------------------------------------------------
    def translate(
        self,
        audio_paths: List[Optional[str]],
        image_paths: List[Optional[str]],
        source_texts: List[str],
        target_lang: str,
        use_cot: bool = False,
        num_beams: int = 5,
        max_new_tokens: int = 256,
    ) -> List[str]:
        """
        Perform batch multimodal translation or speech transcription.
        Accepts any combination of:
            - audio only
            - image + text
            - audio + image
        """

        # Normalize lengths
        max_len = max(len(audio_paths), len(image_paths), len(source_texts))

        audio_paths = audio_paths + [None] * (max_len - len(audio_paths))
        image_paths = image_paths + [None] * (max_len - len(image_paths))
        source_texts = source_texts + [""] * (max_len - len(source_texts))

        # Construct input objects
        inputs = [
            TranslationInput(audio, image, text)
            for audio, image, text in zip(audio_paths, image_paths, source_texts)
        ]

        if not inputs:
            return []

        # Build fused model input tensors
        batch_inputs = self.processor.prepare_batch(
            inputs, target_lang, use_cot
        )
        batch_inputs["beam_size"] = num_beams

        model_eos = self.tokenizer.eos_token_id
        newline_id = self.tokenizer.convert_tokens_to_ids("\n")
        
        # Generate output text
        with torch.inference_mode():
            start = time.time()
            output = self.fuse_model.generate(
                **batch_inputs,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                do_sample=False,
                num_return_sequences=1,
                no_repeat_ngram_size=5,
                # eos_token_id=[model_eos, newline_id], ## In case you want to stop at new line as well
            )
            print(f"Generation time: {time.time() - start:.2f}s")

        # Decode new tokens only
        prefix_len = batch_inputs["input_ids"].shape[1]
        translations = self.tokenizer.batch_decode(
            output[:, prefix_len:], skip_special_tokens=True
        )

        return translations

