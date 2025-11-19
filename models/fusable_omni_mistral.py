from transformers import MistralForCausalLM, MistralPreTrainedModel, MistralModel
import torch
from torch import nn
from typing import Callable, List, Optional, Tuple, Union
from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
    BaseModelOutputWithPast,
)
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.processing_utils import Unpack
#from transformers.models.llama.modeling_llama import KwargsForCausalLM
from transformers.generation import GenerationMixin
from transformers.utils import logging, TransformersKwargs
from transformers import AutoTokenizer
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor, AutoTokenizer
from torch.nn.utils.rnn import pad_sequence
from models.fusion_layer import FusionLayer
from transformers.models.mistral.modeling_mistral import MistralAttention, MistralRMSNorm



logger = logging.get_logger(__name__)

class FusableMistralModel(MistralModel):
    def __init__(self, config, omni_model, omni_embed_dim=3584, depth=3, mode="mid"):
        # Initialize the parent class
        super().__init__(config)
        # You can add any custom layers or modules here if needed
        proj_layers = [nn.Linear(omni_embed_dim, config.hidden_size)]
        for _ in range(1, depth):
            proj_layers.extend([nn.GELU(),
                           nn.Linear(config.hidden_size, config.hidden_size)])
        self.omni_projection = nn.Sequential(*proj_layers)
        #self.omni_qformer = QFormer(num_queries=512, embed_dim=config.hidden_size, num_layers=2, dropout=0.1, num_heads=6)
        self.omni_model = omni_model

        self.omni_fusion = FusionLayer(hidden_dim=omni_embed_dim, mode=mode)

        #self.omni_attention = MistralAttention(config, layer_idx=0)

        #self.omni_input_layernorm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        #self.omni_post_attention_layernorm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        for name, param in self.omni_model.named_parameters():
            param.requires_grad = False

        self.tokenizer = AutoTokenizer.from_pretrained("ByteDance-Seed/Seed-X-PPO-7B", cache_dir="/export/data1/skoneru/hf_cache")


    def left_pad_sequence(self, sequences, batch_first=True, padding_value=0.0):
        """
        Left-pads a list of [seq_len, hidden_dim] tensors to the same length.
        """
        # Find max length
        max_len = max(seq.size(0) for seq in sequences)
        hidden_dim = sequences[0].size(1)

        padded = []
        for seq in sequences:
            pad_len = max_len - seq.size(0)
            # [pad_len, hidden_dim] of padding + original sequence
            pad_tensor = seq.new_full((pad_len, hidden_dim), padding_value)
            padded.append(torch.cat([pad_tensor, seq], dim=0))

        if batch_first:
            return torch.stack(padded, dim=0)  # [batch, max_len, hidden_dim]
        else:
            return torch.stack(padded, dim=1)  # [max_len, batch, hidden_dim]


    def prepare_omni_for_generation(self, omni_embeds, omni_attention_mask, input_ids, attention_mask):
        """
        Prepares the concatenated omni embeddings, input_ids to inputs_embeds, and attention_mask for the model.
        All the inputs are padded individually before being passed to this function.
        Args:
            omni_embeds (torch.FloatTensor): The omni embeddings of shape (batch_size, omni_seq_len, embed_dim).
            input_ids (torch.LongTensor): The input IDs of shape (batch_size, seq_len).
            attention_mask (torch.Tensor): The attention mask of shape (batch_size, seq_len).
        """

        nonpadded_omni_embeds = []
        concat_input_ids = []

        nonpadded_input_ids = []

        pad_token_id = 65269
        placeholder_token_id = 1

        for idx in range(input_ids.size(0)):
            nonpadded_input_id = input_ids[idx, attention_mask[idx].bool()]
            nonpadded_input_ids.append(nonpadded_input_id)

            nonpadded_omni_embeds.append(omni_embeds[idx][ omni_attention_mask[idx].bool().to(omni_embeds.device)])

            concat_ids = nonpadded_input_id
            concat_input_ids.append(concat_ids)


        padded_concat_ids = pad_sequence( concat_input_ids, batch_first=True,
                            padding_value= pad_token_id).to(self.embed_tokens.weight.device)

        concat_attention_mask = (padded_concat_ids != pad_token_id).long()

        with torch.inference_mode():
            batch_embeds = self.embed_tokens(padded_concat_ids)

        text_embeds = [batch_embeds[i, concat_attention_mask[i].bool().to(batch_embeds.device)] for i in range(input_ids.size(0))]

        omni_lengths = [emb.size(0) for emb in nonpadded_omni_embeds]

        omni_placeholders = [
            torch.zeros(w_len, text_embeds[0].shape[-1]) for w_len in omni_lengths
        ]


        concat_embeds = [
            torch.cat((omni_placeholders[i].to(text_embeds[i].device), text_embeds[i]), dim=0)
            for i in range(len(nonpadded_omni_embeds))
        ]

        inputs_embeds = self.left_pad_sequence(concat_embeds, batch_first=True).to(self.omni_model.dtype)

        input_ids = torch.full((inputs_embeds.size(0), inputs_embeds.size(1)), pad_token_id, dtype=torch.long)

        attention_mask = torch.zeros(inputs_embeds.size(0), inputs_embeds.size(1), dtype=torch.long)

        for i, emb in enumerate(concat_embeds):
            attention_mask[i, -emb.size(0):] = 1

        for i, (omni_tokens, text_tokens) in enumerate(zip(omni_placeholders, concat_input_ids)):
            w_len = omni_tokens.size(0)
            l_len = len(text_tokens)
            input_ids[i, :w_len] = 1
            input_ids[i, w_len:w_len + l_len] = torch.tensor(text_tokens, dtype=torch.long)


        return omni_embeds, omni_lengths, inputs_embeds, attention_mask, input_ids

    def prepare_omni_batch(self, omni_embeds, omni_attention_mask, input_ids, attention_mask, text_labels, text_labels_attention_mask):
        """
        Prepares the concatenated omni embeddings, input_ids to inputs_embeds, and attention_mask for the model.
        Creates the aligned labels batch
        All the inputs are padded individually before being passed to this function.
        Args:
            omni_embeds (torch.FloatTensor): The omni embeddings of shape (batch_size, omni_seq_len, embed_dim).
            input_ids (torch.LongTensor): The input IDs of shape (batch_size, seq_len).
            attention_mask (torch.Tensor): The attention mask of shape (batch_size, seq_len).
            labels (torch.LongTensor): The labels of shape (batch_size, seq_len).
            labels_attention_mask (torch.Tensor): The labels attention mask of shape (batch_size, seq_len).
        """

        dummy_nonpadded_omni_ids = []
        nonpadded_omni_embeds = []
        concat_input_label_ids = []

        nonpadded_input_ids = []
        nonpadded_labels = []

        pad_token_id = 65269
        placeholder_token_id = 1

        for idx in range(input_ids.size(0)):
            nonpadded_input_id = input_ids[idx, attention_mask[idx].bool()]
            nonpadded_input_ids.append(nonpadded_input_id)
            nonpadded_label = text_labels[idx, text_labels_attention_mask[idx].bool()]
            nonpadded_labels.append(nonpadded_label)

            dummy_nonpadded_omni_ids.append(torch.zeros(sum(omni_attention_mask[idx] == 1), dtype=torch.long, device=input_ids.device))

            nonpadded_omni_embeds.append(omni_embeds[idx][ omni_attention_mask[idx].bool().to(omni_embeds.device)])

            concat_ids = torch.cat((nonpadded_input_id, nonpadded_label), dim=0)
            concat_input_label_ids.append(concat_ids)

        padded_concat_ids = pad_sequence( concat_input_label_ids, batch_first=True,
                            padding_value= pad_token_id).to(self.embed_tokens.weight.device)

        concat_attention_mask = (padded_concat_ids != pad_token_id).long()

        with torch.inference_mode():
            batch_embeds = self.embed_tokens(padded_concat_ids)

        text_embeds = [batch_embeds[i, concat_attention_mask[i].bool().to(batch_embeds.device)] for i in range(input_ids.size(0))]

        omni_lengths = [emb.size(0) for emb in nonpadded_omni_embeds]

        omni_placeholders = [
            torch.zeros(w_len, text_embeds[0].shape[-1]) for w_len in omni_lengths
        ]


        concat_embeds = [
            torch.cat((omni_placeholders[i].to(text_embeds[i].device), text_embeds[i]), dim=0)
            for i in range(len(nonpadded_omni_embeds))
        ]

        inputs_embeds = pad_sequence(concat_embeds, batch_first=True).to(self.omni_model.dtype)

        input_ids = torch.full((inputs_embeds.size(0), inputs_embeds.size(1)), pad_token_id, dtype=torch.long)
        labels = torch.full((inputs_embeds.size(0), inputs_embeds.size(1)), -100, dtype=torch.long)

        attention_mask = torch.zeros(inputs_embeds.size(0), inputs_embeds.size(1), dtype=torch.long)
        for i, emb in enumerate(concat_embeds):
            attention_mask[i, :emb.size(0)] = 1

        for i, (omni_tokens, text_tokens, label_tokens) in enumerate(zip(omni_placeholders, concat_input_label_ids, nonpadded_labels)):
            w_len = omni_tokens.size(0)
            t_len = len(text_tokens)
            l_len = len(label_tokens)
            input_ids[i, :w_len] = 1
            input_ids[i, w_len:w_len + t_len] = torch.tensor(text_tokens, dtype=torch.long)
            labels[i, w_len + t_len - l_len :w_len + t_len ] = torch.tensor(label_tokens, dtype=torch.long)

        label_lengths = [len(lbl) for lbl in nonpadded_labels]

        # Compute the total input_ids length per example
        input_lengths = [len(l) for l in concat_input_label_ids]

        num_pad_tokens = [input_ids[i].eq(pad_token_id).sum().item() for i in range(len(input_ids))]

        # The position in input_ids where the label sequence *starts*
        label_start_indices_from_end = [-1 * (label_lengths[i] + num_pad_tokens[i]) for i in range(len(nonpadded_labels))]


        max_label_len = min(label_start_indices_from_end) # This is already negative so slice with max_label_len happens from back

        # Now we can safely slice and mask
        # Get the last max_label_len tokens for labels
        labels = labels[:, max_label_len:]

        return omni_embeds, omni_lengths, inputs_embeds, attention_mask, input_ids, labels

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        omni_embeds: Optional[torch.FloatTensor] = None,
        omni_attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if past_key_values is None or past_key_values.get_seq_length() == 0:

            assert omni_embeds is not None
            omni_embeds = omni_embeds.to(inputs_embeds.dtype)

            omni_projected_embeds = self.omni_projection(omni_embeds)
            assert omni_projected_embeds.shape[2] == inputs_embeds.shape[2]

            seq_len = attention_mask.shape[1]
            indices = torch.arange(seq_len).expand_as(attention_mask).to(attention_mask.device)
            masked = attention_mask * indices + (attention_mask == 0) * seq_len
            leading_pad_cnts = masked.min(dim=1).values
            for idx in range(len(omni_attention_mask)):
                inputs_embeds[idx, leading_pad_cnts[idx]: leading_pad_cnts[idx] + torch.sum(omni_attention_mask[idx].bool()), :] = omni_projected_embeds[idx][omni_attention_mask[idx].bool().to(omni_projected_embeds.device)]

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        mask_function = create_causal_mask if self.config.sliding_window is None else create_sliding_window_causal_mask
        causal_mask = mask_function(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds
        if torch.isnan(hidden_states).any():
            print("Nan seen in hidden states")

        # create position embeddings to be shared across the decoder layers
        #position_embeddings = self.rotary_emb(hidden_states, position_ids)
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        cos, sin = cos.to(hidden_states.device), sin.to(hidden_states.device)
        position_embeddings = (cos, sin)

        #if past_key_values is None or past_key_values.get_seq_length() == 0:
            #residual = hidden_states
            #hidden_states = self.omni_input_layernorm(hidden_states)
            #hidden_states, _ = self.omni_attention(hidden_states, position_embeddings, attention_mask=attention_mask.to(hidden_states.device))
            #hidden_states = self.omni_post_attention_layernorm(hidden_states)
            #hidden_states += residual

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )

class FusableMistralForCausalLM(MistralPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config, omni_model, omni_embed_dim, depth, mode="mid"):
        super().__init__(config)
        self.model = FusableMistralModel(config, omni_model, omni_embed_dim, depth, mode)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        def get_input_embeddings(self):
            return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        text_input_ids: torch.LongTensor = None,
        text_attention_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        text_labels_attention_mask: Optional[torch.Tensor] = None,
        # New arguments for omni embeddings
        omni_input_ids: torch.LongTensor = None,
        omni_attention_mask: Optional[torch.Tensor] = None,
        omni_pixel_values: Optional[torch.FloatTensor] = None,
        omni_image_grid_thw: Optional[torch.LongTensor] = None,
        omni_input_features: Optional[torch.FloatTensor] = None,
        omni_feature_attention_mask: Optional[torch.Tensor] = None,
        beam_size: Optional[int] = None, ## Ugly fix for beam search pixel duplication - Ideally code here should not know about generation specifics. We are doing here to know how to split pixel values for each example in batch
        **kwargs: Unpack[TransformersKwargs],

        #**kwargs: Unpack[KwargsForCausalLM],
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            logits_to_keep (`int` or `torch.Tensor`, *optional*):
                If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
                If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
                This is useful when using packed tensor format (single dimension for batch and sequence length).

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, MistralForCausalLM

        >>> model = MistralForCausalLM.from_pretrained("meta-Mistral/Mistral-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-Mistral/Mistral-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        # Reminder to check for pixel values and its shape in first dimension. should be flattened

        if past_key_values is None or past_key_values.get_seq_length() == 0:
            omni_inputs = {
                "input_ids": omni_input_ids,
                "attention_mask": omni_attention_mask,
                "pixel_values": omni_pixel_values if omni_pixel_values is not None else None,
                "image_grid_thw": omni_image_grid_thw,
                "input_features": omni_input_features,
                "feature_attention_mask": omni_feature_attention_mask,
            }
            
            if labels is None and omni_image_grid_thw is not None:
                bsize = omni_input_ids.shape[0]
                assert bsize % beam_size == 0, "Batch size must be multiple of num beams"
                omni_inputs['pixel_values'] =  self.beam_duplicate_pixel_values(omni_pixel_values[0], omni_image_grid_thw[0], beam_size)
                omni_inputs['image_grid_thw'] = omni_image_grid_thw[0].repeat_interleave(beam_size, dim=0)


            omni_inputs = {k: v.to(self.model.omni_model.device) for k, v in omni_inputs.items() if v is not None}
            
            omni_outputs = self.model.omni_model(**omni_inputs, output_hidden_states=True)

            omni_embeds_zero = omni_outputs.hidden_states[0]  # Choose the 6th layer's hidden states as omni embeddings
            omni_embeds_mid = omni_outputs.hidden_states[len(omni_outputs.hidden_states)//2]  # Choose the 6th layer's hidden states as omni embeddings
            omni_embeds_last = omni_outputs.hidden_states[-1]  # Choose the 6th layer's hidden states as omni embeddings

            omni_embeds = self.model.omni_fusion(omni_embeds_zero, omni_embeds_mid, omni_embeds_last)

            del omni_outputs
            
            if labels is not None:
                omni_embeds, omni_lengths, inputs_embeds, attention_mask, input_ids, labels = self.model.prepare_omni_batch(omni_embeds, omni_attention_mask, text_input_ids, text_attention_mask, labels, text_labels_attention_mask)
                input_ids = None
                kwargs.pop("labels", None)
            else:
                # We are in generation mode.
                omni_embeds, omni_lengths, inputs_embeds, attention_mask, input_ids = self.model.prepare_omni_for_generation(omni_embeds, omni_attention_mask, text_input_ids, text_attention_mask)
                kwargs.pop("input_ids", None)
                input_ids = None
            
        else:
            omni_embeds = None
            omni_lengths = None
            input_ids = kwargs.pop("input_ids", None)


        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            text_labels_attention_mask=text_labels_attention_mask,
            # New arguments for omni embeddings
            omni_embeds=omni_embeds,
            omni_attention_mask=omni_attention_mask,
            labels=labels, # THIS IS BAD AS LABELS ARE NOT USED IN MODEL FORWARD, BUT NEEDED FOR PREPARATION IN STREAMING MODE - Implement psuedo label batching in model forward with omni_inputs lengths
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        # The data collator gives you labels in kwargs but we do not want to use that in loss computation, we need to remove from kwargs
        if labels is not None:
            logits_to_keep = labels.shape[1]
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            if torch.isnan(logits).any():
                print("Nan Seen in logits")
                logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
            try:
                assert torch.all((labels == -100) | (labels >= 0)), "Labels tensor contains invalid values"
                assert logits.shape[1] == labels.shape[1]
            except AssertionError as e:
                print(f"Assertion error: {e}")
                print(f"Logits shape: {logits.shape}, Labels shape: {labels.shape}")

            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)


        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def beam_duplicate_pixel_values(self, pixel_values, image_grid_thw, num_beams):

        example_lengths = [t * h * w for (t, h, w) in image_grid_thw]

        split_values = torch.split(pixel_values, example_lengths, dim=0)

        repeated = []

        for ex in split_values:
            repeated.extend([ex] * num_beams)

        duplicated_pixel_values = torch.cat(repeated, dim=0)

        return duplicated_pixel_values
