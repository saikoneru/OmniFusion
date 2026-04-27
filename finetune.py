"""
Minimal fine-tuning script for speech translation using a fused
Omni encoder + LLM decoder architecture.

Usage:
    torchrun --nproc_per_node=<N> train_st.py \
        --st_dataset   /path/to/train.json \
        --valid_dataset /path/to/valid.json \
        --output_dir   /path/to/output \
        --run_name     my_run \
        --fuse_llm     ByteDance-Seed/Seed-X-PPO-7B
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_DISABLE_CODECARBON"] = "1"

import argparse
import warnings
import logging
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

import torch
from datasets import load_dataset, Features, Value
from transformers import (
    AutoTokenizer,
    Qwen2_5OmniProcessor,
    Qwen2_5OmniThinkerForConditionalGeneration,
    TrainingArguments,
    Trainer,
    set_seed,
)
from peft import get_peft_model, LoraConfig

from models.fusable_omni_mistral import FusableMistralForCausalLM
from data_collators.streaming_embed_collator import SpeechTranslationDataCollator


set_seed(42)

# ── Schema ────────────────────────────────────────────────────────────────────

TRANSLATION_FEATURES = Features({
    "translation": {
        "tgt":      Value("string"),
        "audio_fp": Value("string"),
        "image_fp": Value("string"),
        "lang":     Value("string"),
        "src":      Value("string"),
        "tower_hyp": Value("string"),
        "ocr":      Value("string"),
    }
})

# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--st_dataset",    type=str, required=True)
    parser.add_argument("--valid_dataset", type=str, required=True)
    parser.add_argument("--output_dir",    type=str, required=True)
    parser.add_argument("--run_name",      type=str, default="st_run")
    parser.add_argument("--fuse_llm",      type=str, default="ByteDance-Seed/Seed-X-PPO-7B")
    parser.add_argument("--hf_cache",      type=str, default="/export/data2/skoneru/hf_cache")
    parser.add_argument("--omni_model",    type=str, default="Qwen/Qwen2.5-Omni-7B")
    return parser.parse_args()


def add_pad_token(tokenizer, model):
    """Add <pad> token and resize model embeddings to match."""
    n_new = tokenizer.add_special_tokens({"pad_token": "<pad>"})
    model.resize_token_embeddings(len(tokenizer))
    if n_new > 0:
        for emb in (model.get_input_embeddings(), model.get_output_embeddings()):
            w = emb.weight.data
            w[-n_new:] = w[:-n_new].mean(dim=0, keepdim=True)


def get_encoder_modules(model):
    """Return fully-qualified names of projection layers inside the
    vision/audio encoders so they can be excluded from LoRA."""
    suffixes = {"q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"}
    return [
        name for name, _ in model.named_modules()
        if ("VisionEncoder" in name or "AudioEncoder" in name)
        and any(name.endswith(s) for s in suffixes)
    ]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Datasets ──────────────────────────────────────────────────────────────
    print("Loading datasets …")
    load_kw = dict(cache_dir=args.hf_cache, streaming=False)

    train_dataset = (
        load_dataset("json", data_files={"train": args.st_dataset}, **load_kw)["train"]
        .cast(TRANSLATION_FEATURES)
    )
    valid_dataset = (
        load_dataset("json", data_files={"valid": args.valid_dataset}, **load_kw)["valid"]
        .cast(TRANSLATION_FEATURES)
    )

    # ── Models ────────────────────────────────────────────────────────────────
    print("Loading models …")

    omni_model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        args.omni_model,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        cache_dir=args.hf_cache,
        _attn_implementation="flash_attention_2",
    )

    fuse_model = FusableMistralForCausalLM.from_pretrained(
        args.fuse_llm,
        cache_dir=args.hf_cache,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        omni_model=omni_model,
        omni_embed_dim=3584,
        depth=2,
        mode="attention",
    )
    fuse_model.to("cuda:0")
    fuse_model.model.omni_model.to("cuda:0")

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.fuse_llm, cache_dir=args.hf_cache)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        add_pad_token(tokenizer, fuse_model)

    # ── LoRA ──────────────────────────────────────────────────────────────────
    peft_config = LoraConfig(
        inference_mode=False,
        r=16,
        lora_alpha=64,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        exclude_modules=get_encoder_modules(fuse_model),
        modules_to_save=["omni_projection", "omni_fusion"],
    )
    fuse_model = get_peft_model(fuse_model, peft_config)

    # Keep fusion/projection layers trainable
    for name, param in fuse_model.named_parameters():
        if any(k in name for k in ("omni_projection", "omni_gated", "omni_fusion")):
            param.requires_grad = True

    fuse_model.print_trainable_parameters()

    # ── Processor + Collator ──────────────────────────────────────────────────
    omni_processor = Qwen2_5OmniProcessor.from_pretrained(
        args.omni_model, trust_remote_code=True, cache_dir=args.hf_cache
    )
    omni_processor.tokenizer.padding_side = "right"

    collator = SpeechTranslationDataCollator(
        tokenizer=tokenizer,
        omni_processor=omni_processor,
    )

    # ── Training ──────────────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        run_name=args.run_name,
        # Checkpointing / evaluation
        save_strategy="steps",
        eval_strategy="steps",
        save_steps=1000,
        eval_steps=1000,
        save_total_limit=3,
        metric_for_best_model="loss",
        # Logging
        logging_strategy="steps",
        logging_steps=5,
        logging_dir=f"{args.output_dir}/logs",
        report_to="wandb",
        # Optimisation
        learning_rate=1e-4,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        weight_decay=0.01,
        max_steps=20000,
        bf16=True,
        # Gradient checkpointing
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # Misc
        remove_unused_columns=False,
        label_names=["labels"],
        dataloader_num_workers=1,
        ddp_find_unused_parameters=True,
    )

    trainer = Trainer(
        model=fuse_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    trainer.train()


if __name__ == "__main__":
    main()
