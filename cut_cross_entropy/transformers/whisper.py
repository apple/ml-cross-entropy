# Copyright (C) 2024 Apple Inc. All Rights Reserved.
from types import MethodType
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import transformers
from torch.nn import CrossEntropyLoss
from transformers.cache_utils import Cache, EncoderDecoderCache
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.models.whisper.modeling_whisper import (
    _CONFIG_FOR_DOC,
    WHISPER_START_DOCSTRING,
    logger,
)
from transformers.utils import (
    add_start_docstrings_to_model_forward,
    is_torchdynamo_compiling,
    replace_return_docstrings,
)

from cut_cross_entropy import linear_cross_entropy

from .utils import PatchOptions, TransformersModelT

_PATCH_OPTS: PatchOptions | None = None


@add_start_docstrings_to_model_forward(WHISPER_START_DOCSTRING)
@replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
def cce_forward(
    self,
    input_features: Optional[torch.FloatTensor] = None,
    attention_mask: Optional[torch.LongTensor] = None,
    decoder_input_ids: Optional[torch.LongTensor] = None,
    decoder_attention_mask: Optional[torch.LongTensor] = None,
    head_mask: Optional[torch.Tensor] = None,
    decoder_head_mask: Optional[torch.Tensor] = None,
    cross_attn_head_mask: Optional[torch.Tensor] = None,
    encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
    past_key_values: Optional[Union[EncoderDecoderCache, Tuple[torch.FloatTensor]]] = None,
    decoder_inputs_embeds: Optional[Tuple[torch.FloatTensor]] = None,
    decoder_position_ids: Optional[Tuple[torch.LongTensor]] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
) -> Union[Tuple[torch.Tensor], Seq2SeqLMOutput]:
    r"""
    labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
        Labels for computing the language modeling loss. Indices should either be in `[0, ..., config.vocab_size]`
        or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored (masked), the loss is
        only computed for the tokens with labels in `[0, ..., config.vocab_size]`. `sequence_length` should be smaller than or equal to `config.max_target_positions`.

    Returns:

    Example:

    ```python
    >>> import torch
    >>> from transformers import AutoProcessor, WhisperForConditionalGeneration
    >>> from datasets import load_dataset

    >>> processor = AutoProcessor.from_pretrained("openai/whisper-tiny.en")
    >>> model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")

    >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

    >>> inputs = processor(ds[0]["audio"]["array"], return_tensors="pt")
    >>> input_features = inputs.input_features

    >>> generated_ids = model.generate(inputs=input_features)

    >>> transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    >>> transcription
    ' Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.'
    ```"""
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if labels is not None:
        if labels.shape[1] > self.max_target_positions:
            raise ValueError(
                f"Labels' sequence length {labels.shape[1]} cannot exceed the maximum allowed length of {self.max_target_positions} tokens."
            )
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = shift_tokens_right(
                labels, self.config.pad_token_id, self.config.decoder_start_token_id
            )

    outputs = self.model(
        input_features,
        attention_mask=attention_mask,
        decoder_input_ids=decoder_input_ids,
        encoder_outputs=encoder_outputs,
        decoder_attention_mask=decoder_attention_mask,
        head_mask=head_mask,
        decoder_head_mask=decoder_head_mask,
        cross_attn_head_mask=cross_attn_head_mask,
        past_key_values=past_key_values,
        decoder_inputs_embeds=decoder_inputs_embeds,
        decoder_position_ids=decoder_position_ids,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
    )
    hidden_states = outputs[0]
    hidden_states = hidden_states.type(self.proj_out.weight.dtype)

    loss = None
    lm_logits = None

    if labels is not None and _PATCH_OPTS is not None:
        loss = linear_cross_entropy(
            hidden_states,
            self.proj_out.weight,
            labels.to(hidden_states.device),
            shift=False,
            impl=_PATCH_OPTS.impl,
            reduction=_PATCH_OPTS.reduction,
        )

    if not return_dict:
        output = (lm_logits,) + outputs[1:]
        return ((loss,) + output) if loss is not None else output

    return Seq2SeqLMOutput(
        loss=loss,
        logits=lm_logits,
        past_key_values=outputs.past_key_values,
        decoder_hidden_states=outputs.decoder_hidden_states,
        decoder_attentions=outputs.decoder_attentions,
        cross_attentions=outputs.cross_attentions,
        encoder_last_hidden_state=outputs.encoder_last_hidden_state,
        encoder_hidden_states=outputs.encoder_hidden_states,
        encoder_attentions=outputs.encoder_attentions,
    )


def patch_whisper(
    maybe_model: TransformersModelT | str | transformers.PretrainedConfig,
    patch_options: PatchOptions,
) -> TransformersModelT | None:
    global _PATCH_OPTS
    from transformers.models.whisper import modeling_whisper

    _PATCH_OPTS = patch_options

    if isinstance(maybe_model, transformers.PreTrainedModel):
        assert isinstance(
            maybe_model, modeling_whisper.WhisperForConditionalGeneration
        ), f"Expected a WhisperForConditionalGeneration model. Got {type(maybe_model)}."
        maybe_model.forward = MethodType(cce_forward, maybe_model)
        return maybe_model
    else:
        modeling_whisper.WhisperForConditionalGeneration.forward = cce_forward
