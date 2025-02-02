import logging
import warnings

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from model.LMConfig import LMConfig
from model.model import Transformer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(name)s %(levelname)s %(message)s')

warnings.filterwarnings('ignore', category=UserWarning)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def export_transformers_model():
    LMConfig.register_for_auto_class()
    Transformer.register_for_auto_class("AutoModelForCausalLM")

    lm_config = LMConfig()
    lm_model = Transformer(lm_config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    moe_path = '_moe' if lm_config.use_moe else ''
    # ckpt_path = f'./out/single_chat/full_sft_{lm_config.dim}{moe_path}.pth'
    ckpt_path = f'./out/full_sft_{lm_config.dim}{moe_path}.pth'

    state_dict = torch.load(ckpt_path, map_location=device)
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    lm_model.load_state_dict(state_dict, strict=False)
    print(f'模型参数: {count_parameters(lm_model) / 1e6} 百万 = {count_parameters(lm_model) / 1e9} B (Billion)')

    # save model to local ./minimind-v1-small directory
    lm_model.save_pretrained("./save_models/minimind-v1-small", safe_serialization=False)


def export_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained('./model/minimind_tokenizer',
                                              trust_remote_code=True, use_fast=False)
    # save tokenizer to local ./minimind-v1-small directory
    tokenizer.save_pretrained("./save_models/minimind-v1-small")


def push_to_hf():
    def init_model():
        tokenizer = AutoTokenizer.from_pretrained('./model/minimind_tokenizer',
                                                  trust_remote_code=True, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained('./save_models/minimind-v1-small', trust_remote_code=True)
        return model, tokenizer

    model, tokenizer = init_model()
    # 推送到huggingface
    model.push_to_hub("tclh123/minimind-v1-small", safe_serialization=False)
    # tokenizer.push_to_hub("minimind-v1-small", safe_serialization=False)


if __name__ == '__main__':
    # 1
    export_transformers_model()
    # 2
    export_tokenizer()
    # 3
    push_to_hf()
