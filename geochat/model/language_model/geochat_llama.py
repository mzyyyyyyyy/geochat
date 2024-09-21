#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast

from ..geochat_arch import GeoChatMetaModel, GeoChatMetaForCausalLM


class GeoChatConfig(LlamaConfig):
    model_type = "geochat"


class GeoChatLlamaModel(GeoChatMetaModel, LlamaModel):
    config_class = GeoChatConfig

    def __init__(self, config: LlamaConfig):
        print(GeoChatLlamaModel.__mro__)
        super(GeoChatLlamaModel, self).__init__(config)
        # super() 函数用于调用父类的构造函数。这行代码调用了父类 GeoChatMetaModel 和 LlamaModel 的构造函数，并将 config 参数传递给它们。
        # 这保证了父类的初始化逻辑能够正常执行，父类的属性和状态会在 GeoChatLlamaModel 的实例中正确设置。


class GeoChatLlamaForCausalLM(LlamaForCausalLM, GeoChatMetaForCausalLM): 
# 这个类从两个父类继承，通过这种继承结构，该类可以结合这两个父类的功能，实现更复杂的行为和任务。
    config_class = GeoChatConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = GeoChatLlamaModel(config)
        # 这个 GeoChatLlamaModel 类定义了 GeoChat 模型的架构，即，可以通过该类提取特征，但不能实现功能。
        # 而该类 GeoChatLlamaForCausalLM 实现的是在 GeoChat 模型的架构基础上，增加了因果语言建模(Causal Language Modeling)功能，如文本生成等。

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        input_ids, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, images) 
        # attention_mask.shape = torch.Size([1, 1357]), inputs_embeds = torch.Size([1, 1357, 4096])。1357 = 36 * 36 + 62 - 1 . 
        # 第一次进来的时候 attention_mask = (1, 62), 第二次进来的时候 attention_mask = (1, 63). images 没有变。

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn) 
        # 输入给 model 的 inputs_embeds 相当于用图像编码器编码后的图像信息代替了 prompt_token 中的 image_token，然后将其余 prompt_token 用 embedding 转换为向量。
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        hidden_states = outputs[0] # torch.Size([1, 1357, 4096]) # torch.Size([1, 1, 4096])
        logits = self.lm_head(hidden_states) # torch.Size([1, 1357, 32000]) # torch.Size([1, 1, 32000])

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
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

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
            }
        )
        return model_inputs

AutoConfig.register("geochat", GeoChatConfig)
AutoModelForCausalLM.register(GeoChatConfig, GeoChatLlamaForCausalLM)
