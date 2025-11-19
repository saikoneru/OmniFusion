import gradio as gr
from models.fusable_omni_mistral import FusableMistralForCausalLM
from transformers import AutoTokenizer, Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration
import torch
from typing import Dict, List, Optional, Tuple
import transformers
from qwen_omni_utils import process_mm_info
from PIL import Image
import time
from peft import PeftModel
from dataclasses import dataclass


LANG_MAP = {
    "Chinese": "zh", "English": "en", "French": "fr", "German": "de",
    "Italian": "it", "Japanese": "ja", "Korean": "ko", "Portuguese": "pt",
    "Russian": "ru", "Spanish": "es", "Vietnamese": "vi", "Arabic": "ar",
    "Czech": "cs", "Croatian": "hr", "Danish": "da", "Dutch": "nl",
    "Finnish": "fi", "Hungarian": "hu", "Indonesian": "id", "Malay": "ms",
    "Norwegian Bokmal": "nb", "Norwegian": "no", "Polish": "pl",
    "Romanian": "ro", "Turkish": "tr"
}


@dataclass
class TranslationInput:
    """Container for a single translation input."""
    audio_path: Optional[str] = None
    image_path: Optional[str] = None
    source_text: str = ""


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding."""
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


class OmniFusionProcessor:
    """Processor for combining Omni and Fusion model inputs."""
    
    SYSTEM_PROMPT = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."
    IMAGE_SIZE = (512, 512)
    USE_AUDIO_IN_VIDEO = True
    
    def __init__(
        self,
        omni_processor: Qwen2_5OmniProcessor,
        text_tokenizer: transformers.PreTrainedTokenizer,
        device: torch.device
    ):
        self.omni_processor = omni_processor
        self.text_tokenizer = text_tokenizer
        self.device = device
        
    def _build_conversation(
        self,
        translation_input: TranslationInput,
        target_lang: str,
        use_cot: bool = False
    ) -> Tuple[List[Dict], str]:
        """Build conversation and text input for a single item."""
        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.SYSTEM_PROMPT}],
            },
        ]
        
        lang_tag = LANG_MAP[target_lang]
        prefix_tower = ""
        suffix_tower = f" <{lang_tag}> "
        
        # Audio present (with or without image)
        if translation_input.audio_path is not None:
            content = [
                {"type": "text", "text": "Transcribe the audio with using the image for context along with OCR on it for spelling keywords and names:"}
            ]
            
            if translation_input.image_path is not None:
                content.append({"type": "image", "image": translation_input.image_path})
            
            content.append({"type": "audio", "audio": translation_input.audio_path})
            
            conversation.append({"role": "user", "content": content})
            
            # Apply CoT if requested and source text exists
            if use_cot and translation_input.source_text:
                suffix_tower = translation_input.source_text + suffix_tower
                
        # Image only (no audio) with source text
        elif translation_input.image_path is not None:
            if translation_input.source_text == "":
                suffix_tower = " <OCR> "
            else:
                prefix_tower = f" <SRC> {translation_input.source_text}{self.text_tokenizer.eos_token}"
            
            conversation.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Translate the following sentence into {target_lang} using the image for context. <IMG> refers to the image. Use the image for formality, gender, keywords context and other details important for disambiguation:\n{translation_input.source_text} <IMG>"
                    },
                    {"type": "image", "image": translation_input.image_path},
                ],
            })
        
        text_input = prefix_tower + suffix_tower
        return conversation, text_input
    
    def _resize_images(self, images: Optional[List]) -> Optional[List]:
        """Resize images to standard size."""
        if images is None:
            return None
        return [img.resize(self.IMAGE_SIZE) if img is not None else None for img in images]
    
    def prepare_batch(
        self,
        inputs: List[TranslationInput],
        target_lang: str,
        use_cot: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Prepare batch inputs for the fusion model."""
        conversations = []
        text_inputs = []
        
        # Build conversations and text inputs
        for translation_input in inputs:
            conversation, text_input = self._build_conversation(
                translation_input, target_lang, use_cot
            )
            conversations.append(conversation)
            text_inputs.append(text_input)
        
        # Process with Omni processor
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
            use_audio_in_video=self.USE_AUDIO_IN_VIDEO
        ).to(self.device)
        
        # Process with text tokenizer
        text_inputs_tokenized = self.text_tokenizer(
            text_inputs,
            max_length=512,
            return_tensors='pt',
            add_special_tokens=False,
            padding="longest"
        ).to(self.device)
        
        # Concatenate input IDs
        concat_input_ids = self._concatenate_input_ids(omni_inputs, text_inputs_tokenized)
        
        # Build final inputs dictionary
        return self._build_final_inputs(omni_inputs, text_inputs_tokenized, concat_input_ids)
    
    def _concatenate_input_ids(
        self,
        omni_inputs,
        text_inputs
    ) -> torch.Tensor:
        """Concatenate Omni and text input IDs."""
        omni_input_ids_nopad = [
            seq[seq != self.omni_processor.tokenizer.pad_token_id]
            for seq in omni_inputs.input_ids
        ]
        text_input_ids_nopad = [
            seq[seq != self.text_tokenizer.pad_token_id]
            for seq in text_inputs.input_ids
        ]
        
        concat_input_ids = [
            torch.cat([o, t], dim=0)
            for o, t in zip(omni_input_ids_nopad, text_input_ids_nopad)
        ]
        
        # Pad to max length
        max_len = max(seq.size(0) for seq in concat_input_ids)
        padded_input_ids = torch.full(
            (len(concat_input_ids), max_len),
            self.text_tokenizer.pad_token_id,
            dtype=concat_input_ids[0].dtype,
            device=concat_input_ids[0].device
        )
        
        for i, seq in enumerate(concat_input_ids):
            padded_input_ids[i, -seq.size(0):] = seq
        
        return padded_input_ids
    
    def _build_final_inputs(
        self,
        omni_inputs,
        text_inputs,
        concat_input_ids: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Build the final inputs dictionary."""
        return {
            "omni_input_ids": omni_inputs.input_ids,
            "omni_attention_mask": omni_inputs.attention_mask,
            "omni_pixel_values": [omni_inputs.pixel_values] if "pixel_values" in omni_inputs else None,
            "omni_image_grid_thw": [omni_inputs.image_grid_thw] if "image_grid_thw" in omni_inputs else None,
            "omni_input_features": omni_inputs.input_features if "input_features" in omni_inputs else None,
            "omni_feature_attention_mask": omni_inputs.feature_attention_mask if "feature_attention_mask" in omni_inputs else None,
            "text_input_ids": text_inputs.input_ids,
            "text_attention_mask": text_inputs.attention_mask,
            "input_ids": concat_input_ids,
            "attention_mask": (concat_input_ids != self.text_tokenizer.pad_token_id).long(),
        }


class MultimodalTranslationSystem:
    """Main translation system combining Omni and Fusion models."""
    
    def __init__(
        self,
        cache_dir: str = "",
        checkpoint_path: str = "skoneru/OmniFusion"
    ):
        self.cache_dir = cache_dir
        self.checkpoint_path = checkpoint_path
        self.device = torch.device("cuda:0")
        
        print("Loading models...")
        self._load_models()
        self._setup_processor()
        print("Models loaded successfully!")
    
    def _load_models(self):
        """Load all required models."""
        # Load Omni model
        self.omni_model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-Omni-7B",
            dtype=torch.bfloat16,
            trust_remote_code=True,
            cache_dir=self.cache_dir,
            device_map="cuda:0",
        )
        
        # Load Omni processor
        self.omni_processor = Qwen2_5OmniProcessor.from_pretrained(
            "Qwen/Qwen2.5-Omni-7B",
            trust_remote_code=True,
            cache_dir=self.cache_dir
        )
        self.omni_processor.tokenizer.padding_side = "left"
        
        # Load text tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            "ByteDance-Seed/Seed-X-PPO-7B",
            cache_dir=self.cache_dir
        )
        self.tokenizer.padding_side = "left"
        
        # Load fusion model
        self.fuse_model = FusableMistralForCausalLM.from_pretrained(
            "ByteDance-Seed/Seed-X-PPO-7B",
            cache_dir=self.cache_dir,
            device_map="cuda:0",
            torch_dtype=torch.bfloat16,
            omni_model=self.omni_model,
            omni_embed_dim=3584,
            depth=2,
            mode="gated"
        )
        
        # Add pad token if needed
        if self.tokenizer.pad_token is None:
            print("Adding pad token as '<pad>'")
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="<pad>"),
                tokenizer=self.tokenizer,
                model=self.fuse_model,
            )
        
        # Load PEFT model and merge
        self.fuse_model = PeftModel.from_pretrained(
            self.fuse_model,
            self.checkpoint_path
        )
        
        self.fuse_model = self.fuse_model.merge_and_unload()
        
        # Freeze parameters
        for param in self.fuse_model.parameters():
            param.requires_grad = False
    
    def _setup_processor(self):
        """Setup the fusion processor."""
        self.processor = OmniFusionProcessor(
            omni_processor=self.omni_processor,
            text_tokenizer=self.tokenizer,
            device=self.device
        )
    
    def translate(
        self,
        audio_paths: List[Optional[str]],
        image_paths: List[Optional[str]],
        source_texts: List[str],
        target_lang: str,
        use_cot: bool = False,
        num_beams: int = 5,
        max_new_tokens: int = 256
    ) -> List[str]:
        """
        Translate a batch of inputs.
        
        Args:
            audio_paths: List of audio file paths (can be None)
            image_paths: List of image file paths (can be None)
            source_texts: List of source texts
            target_lang: Target language
            use_cot: Chain of thought flag
            num_beams: Number of beams for beam search
            max_new_tokens: Maximum new tokens to generate
            
        Returns:
            List of translations
        """
        # Ensure all lists have the same length
        max_len = max(len(audio_paths), len(image_paths), len(source_texts))
        audio_paths = audio_paths + [None] * (max_len - len(audio_paths))
        image_paths = image_paths + [None] * (max_len - len(image_paths))
        source_texts = source_texts + [""] * (max_len - len(source_texts))
        
        # Build input list
        inputs = [
            TranslationInput(audio, image, text)
            for audio, image, text in zip(audio_paths, image_paths, source_texts)
        ]
        
        if not inputs:
            return []
        
        # Prepare batch
        batch_inputs = self.processor.prepare_batch(inputs, target_lang, use_cot)
        batch_inputs["beam_size"] = num_beams
        
        # Generate
        with torch.inference_mode():
            start_time = time.time()
            output = self.fuse_model.generate(
                **batch_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=num_beams,
                num_return_sequences=1,
                no_repeat_ngram_size=5,
            )
            end_time = time.time()
            print(f"Batch generation time: {end_time - start_time:.2f}s")
        
        # Decode
        translations = self.tokenizer.batch_decode(
            output[:, batch_inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        # Log results
        for i, (src, trans) in enumerate(zip(source_texts, translations)):
            print(f"SRC {i+1}: {src}")
            print(f"Translation {i+1}: {trans}")
        
        return translations
    



def process_single_input(
    audio: Optional[str],
    image: Optional[str],
    text: str,
    lang: str,
    cot: bool,
    system: MultimodalTranslationSystem
) -> str:
    """Helper for single input processing."""
    translations = system.translate(
        audio_paths=[audio] if audio else [],
        image_paths=[image] if image else [],
        source_texts=[text],
        target_lang=lang,
        use_cot=cot
    )
    return translations[0] if translations else ""


def process_batch_inputs(
    audio1: Optional[str],
    image1: Optional[str],
    src1: str,
    audio2: Optional[str],
    image2: Optional[str],
    src2: str,
    target_lang: str,
    use_cot: bool,
    system: MultimodalTranslationSystem
) -> Tuple[str, str]:
    """Helper function to process two inputs for Gradio."""
    audio_paths = []
    image_paths = []
    source_texts = []
    
    # Add first input if any data provided
    if audio1 or image1 or src1:
        audio_paths.append(audio1)
        image_paths.append(image1)
        source_texts.append(src1 or "")
    
    # Add second input if any data provided
    if audio2 or image2 or src2:
        audio_paths.append(audio2)
        image_paths.append(image2)
        source_texts.append(src2 or "")
    
    if not source_texts:
        return "No input provided", ""
    
    translations = system.translate(
        audio_paths=audio_paths,
        image_paths=image_paths,
        source_texts=source_texts,
        target_lang=target_lang,
        use_cot=use_cot
    )
    
    output1 = translations[0] if len(translations) > 0 else ""
    output2 = translations[1] if len(translations) > 1 else ""
    
    return output1, output2


def create_gradio_interface(system: MultimodalTranslationSystem) -> gr.Blocks:
    """Create the Gradio interface."""
    with gr.Blocks(theme="compact") as demo:
        gr.Markdown("# Multimodal Translation System")
        gr.Markdown("Process 1 or 2 inputs in batch. Audio = transcription, Image+Text = translation with context")
        
        with gr.Tab("Single Input"):
            with gr.Row():
                with gr.Column():
                    single_audio = gr.Audio(type="filepath", label="Audio (optional)")
                    single_image = gr.Image(type="filepath", label="Image (optional)")
                    single_text = gr.Textbox(label="Source Text (only for image translation)")
                    single_lang = gr.Dropdown(choices=list(LANG_MAP.keys()), label="Target Language", value="English")
                    single_cot = gr.Checkbox(label="Chain of Thought", value=False)
                    single_btn = gr.Button("Translate")
                
                with gr.Column():
                    single_output = gr.Textbox(label="Translation")
            
            single_btn.click(
                fn=lambda *args: process_single_input(*args, system=system),
                inputs=[single_audio, single_image, single_text, single_lang, single_cot],
                outputs=single_output
            )
        
        with gr.Tab("Batch (2 Inputs)"):
            gr.Markdown("Process two inputs in a single batch for faster processing")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("**Input 1**")
                    batch_audio1 = gr.Audio(type="filepath", label="Audio 1")
                    batch_image1 = gr.Image(type="filepath", label="Image 1")
                    batch_text1 = gr.Textbox(lines=3, label="Source Text 1")
                
                with gr.Column():
                    gr.Markdown("**Input 2**")
                    batch_audio2 = gr.Audio(type="filepath", label="Audio 2")
                    batch_image2 = gr.Image(type="filepath", label="Image 2")
                    batch_text2 = gr.Textbox(lines=3, label="Source Text 2")
            
            with gr.Row():
                batch_lang = gr.Dropdown(choices=list(LANG_MAP.keys()), label="Target Language", value="English")
                batch_cot = gr.Checkbox(label="Chain of Thought", value=False)
                batch_btn = gr.Button("Process Batch")
            
            with gr.Row():
                with gr.Column():
                    batch_output1 = gr.Textbox(lines=8, label="Output 1")
                with gr.Column():
                    batch_output2 = gr.Textbox(lines=8, label="Output 2")
            
            batch_btn.click(
                fn=lambda *args: process_batch_inputs(*args, system=system),
                inputs=[batch_audio1, batch_image1, batch_text1, batch_audio2, batch_image2, batch_text2, batch_lang, batch_cot],
                outputs=[batch_output1, batch_output2]
            )
    
    return demo


if __name__ == "__main__":
    # Initialize the translation system
    system = MultimodalTranslationSystem()
    
    # Create and launch interface
    demo = create_gradio_interface(system)
    demo.launch(server_name="0.0.0.0", server_port=7979)