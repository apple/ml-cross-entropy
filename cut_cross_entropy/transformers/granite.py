from types import MethodType
from typing import List, Optional, Tuple, Union

import torch
import transformers
from torch.nn import CrossEntropyLoss
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import (
    add_start_docstrings_to_model_forward,
    is_torchdynamo_compiling,
    replace_return_docstrings,
)

from cut_cross_entropy import linear_cross_entropy
from .utils import PatchOptions, TransformersModelT

_PATCH_OPTS: PatchOptions | None = None

def cce_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    num_logits_to_keep: int = 0,
) -> Union[Tuple, CausalLMOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
    )

    hidden_states = outputs[0]
    original_dtype = hidden_states.dtype
    loss = None
    logits = None

    if labels is not None and _PATCH_OPTS is not None:
        # Granite uses logit scaling
        if hasattr(self.config, 'logits_scaling'):
            scaling = torch.tensor(self.config.logits_scaling, dtype=original_dtype, device=hidden_states.device)
            hidden_states = hidden_states / scaling
        loss = linear_cross_entropy(
            hidden_states,
            self.lm_head.weight,
            labels.to(hidden_states.device),
            shift=True,
            impl=_PATCH_OPTS.impl,
            reduction=_PATCH_OPTS.reduction,
        )
    else:
        # Granite uses logit scaling
        logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])
        if hasattr(self.config, 'logits_scaling'):
            scaling = torch.tensor(self.config.logits_scaling, dtype=original_dtype, device=hidden_states.device)
            hidden_states = hidden_states / scaling
        logits = logits.float()

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )

def patch_granite(
    maybe_model: TransformersModelT | str | transformers.PretrainedConfig,
    patch_options: PatchOptions,
) -> TransformersModelT | None:
    global _PATCH_OPTS
    from transformers.models.granite import modeling_granite

    _PATCH_OPTS = patch_options

    if isinstance(maybe_model, transformers.PreTrainedModel):
        assert isinstance(
            maybe_model, modeling_granite.GraniteForCausalLM
        ), f"Expected a GraniteForCausalLM model. Got {type(maybe_model)}."
        maybe_model.forward = MethodType(cce_forward, maybe_model)
        return maybe_model
    else:
        modeling_granite.GraniteForCausalLM.forward = cce_forward