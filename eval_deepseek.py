import os
import sys
import logging
import argparse
import random
import time

import numpy as np
import torch
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(name)s %(levelname)s %(message)s')

logger.info('Imporing: from model.model import Transformer')
from model.model import Transformer
logger.info('Imporing: from model.LMConfig import LMConfig')
from model.LMConfig import LMConfig

warnings.filterwarnings('ignore')

# "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
# 大概需要使用 11GiB 显存

# TODO:
# The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
# Setting `pad_token_id` to `eos_token_id`:151643 for open-end generation.
# The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.

# Traceback (most recent call last):
#   File "/root/projects/minimind/eval_deepseek.py", line 164, in <module>
#     res_y = model.generate(x, max_new_tokens=max_seq_len, temperature=temperature,
#             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/root/miniconda3/lib/python3.12/site-packages/torch/utils/_contextlib.py", line 115, in decorate_context
#     return func(*args, **kwargs)
#            ^^^^^^^^^^^^^^^^^^^^^
#   File "/root/miniconda3/lib/python3.12/site-packages/transformers/generation/utils.py", line 1689, in generate
#     self._validate_model_kwargs(model_kwargs.copy())
#   File "/root/miniconda3/lib/python3.12/site-packages/transformers/generation/utils.py", line 1243, in _validate_model_kwargs
#     raise ValueError(
# ValueError: The following `model_kwargs` are not used by the model: ['stream'] (note: typos in the generate arguments will also show up in this list)

# Traceback (most recent call last):
#   File "/root/projects/minimind/eval_deepseek.py", line 157, in <module>
#     print(query(prompt_datas[0]))
#           ^^^^^^^^^^^^^^^^^^^^^^
#   File "/root/projects/minimind/eval_deepseek.py", line 86, in query
#     pipe = pipeline("text-generation", model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", device=args.device)
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/root/miniconda3/lib/python3.12/site-packages/transformers/pipelines/__init__.py", line 1097, in pipeline
#     return pipeline_class(model=model, framework=framework, task=task, **kwargs)
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/root/miniconda3/lib/python3.12/site-packages/transformers/pipelines/text_generation.py", line 96, in __init__
#     super().__init__(*args, **kwargs)
#   File "/root/miniconda3/lib/python3.12/site-packages/transformers/pipelines/base.py", line 897, in __init__
#     self.model.to(self.device)
#   File "/root/miniconda3/lib/python3.12/site-packages/transformers/modeling_utils.py", line 2883, in to
#     return super().to(*args, **kwargs)
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/root/miniconda3/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1173, in to
#     return self._apply(convert)
#            ^^^^^^^^^^^^^^^^^^^^
#   File "/root/miniconda3/lib/python3.12/site-packages/torch/nn/modules/module.py", line 779, in _apply
#     module._apply(fn)
#   File "/root/miniconda3/lib/python3.12/site-packages/torch/nn/modules/module.py", line 779, in _apply
#     module._apply(fn)
#   File "/root/miniconda3/lib/python3.12/site-packages/torch/nn/modules/module.py", line 779, in _apply
#     module._apply(fn)
#   [Previous line repeated 2 more times]
#   File "/root/miniconda3/lib/python3.12/site-packages/torch/nn/modules/module.py", line 804, in _apply
#     param_applied = fn(param)
#                     ^^^^^^^^^
#   File "/root/miniconda3/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1159, in convert
#     return t.to(
#            ^^^^^
# torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 54.00 MiB. GPU


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def init_model(lm_config):
    # tokenizer = AutoTokenizer.from_pretrained('./model/minimind_tokenizer')
    # # model_from = 1  # 1从权重，2用transformers

    # if not args.from_transformers:
    #     # moe_path = '_moe' if lm_config.use_moe else ''
    #     # ckp = f'./out/pretrain_{lm_config.dim}{moe_path}.pth'
    #     ckp = os.path.join(args.out_dir, args.checkpoint)
    #     logger.info('Load from checkpoint %s', ckp)

    #     model = Transformer(lm_config)
    #     state_dict = torch.load(ckp, map_location=device)

    #     # 处理不需要的前缀
    #     unwanted_prefix = '_orig_mod.'
    #     for k, v in list(state_dict.items()):
    #         if k.startswith(unwanted_prefix):
    #             state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    #     for k, v in list(state_dict.items()):
    #         if 'mask' in k:
    #             del state_dict[k]

    #     # 加载到模型中
    #     model.load_state_dict(state_dict, strict=False)
    # else:
    #     logger.info('Load from transformers %s', 'minimind')
    #     model = AutoModelForCausalLM.from_pretrained('jingyaogong/minimind-v1-small', trust_remote_code=True)

    # Load model directly
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    model = model.to(device)

    print(f'模型参数: {count_parameters(model) / 1e6} 百万 = {count_parameters(model) / 1e9} B (Billion)')
    return model, tokenizer


def setup_seed(seed):
    random.seed(seed)  # 设置 Python 的随机种子
    np.random.seed(seed)  # 设置 NumPy 的随机种子
    torch.manual_seed(seed)  # 设置 PyTorch 的随机种子
    torch.cuda.manual_seed(seed)  # 为当前 GPU 设置随机种子（如果有）
    torch.cuda.manual_seed_all(seed)  # 为所有 GPU 设置随机种子（如果有）
    torch.backends.cudnn.deterministic = True  # 确保每次返回的卷积算法是确定的
    torch.backends.cudnn.benchmark = False  # 关闭 cuDNN 的自动调优，避免不确定性


# Use a pipeline as a high-level helper
from transformers import pipeline


def query(prompt):
    messages = [
                {"role": "user", "content": prompt},
                ]
    pipe = pipeline("text-generation", model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", device=args.device)
    return pipe(messages)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind eval_pretrain")
    parser.add_argument("--out_dir", type=str, default="out", help="Output directory")
    parser.add_argument("--checkpoint", '-p', type=str, default="pretrain_512-202501280340.pth", help="the checkpoint filename to load")
    parser.add_argument("--from-transformers", '-t', action='store_true', default=False, help="whether to load from transformers")
    parser.add_argument("--auto", '-a', action='store_true', default=False, help="whether to auto test")

    # parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    # parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    # parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu",
                        help="Device to use")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Data type")
    # parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases")
    # parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain", help="Weights & Biases project name")
    # parser.add_argument("--num_workers", type=int, default=1, help="Number of workers for data loading")
    # parser.add_argument("--data_path", type=str, default="./dataset/pretrain_data.csv", help="Path to training data")
    # parser.add_argument("--ddp", action="store_true", help="Use DistributedDataParallel")
    # parser.add_argument("--accumulation_steps", type=int, default=8, help="Gradient accumulation steps")
    # parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping threshold")
    # parser.add_argument("--warmup_iters", type=int, default=0, help="Number of warmup iterations")
    # parser.add_argument("--log_interval", type=int, default=100, help="Logging interval")
    # parser.add_argument("--save_interval", type=int, default=1000, help="Model saving interval")
    # parser.add_argument('--local_rank', type=int, default=-1, help='local rank for distributed training')

    args = parser.parse_args()
    logger.info('Run with args: %s', args)

    # -----------------------------------------------------------------------------
    out_dir = args.out_dir
    start = ""
    temperature = 0.7
    top_k = 8
    setup_seed(1337)
    # device = 'cpu'
    # device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = args.device
    # dtype = 'bfloat16'
    dtype = args.dtype
    max_seq_len = 512
    lm_config = LMConfig()
    lm_config.max_seq_len = max_seq_len
    # -----------------------------------------------------------------------------

    model, tokenizer = init_model(lm_config)
    model = model.eval()
    # int(input('输入0自动测试，输入1问题测试：'))
    answer_way = 'auto' if args.auto else 'user'
    stream = True

    prompt_datas = [
        '椭圆和圆的区别',
        '中国关于马克思主义基本原理',
        '人类大脑的主要功能是',
        '万有引力是',
        '世界上人口最多的国家是',
        'DNA的全称是',
        '数学中π的值大约是',
        '世界上最高的山峰是',
        '太阳系中最大的行星是',
        '二氧化碳的化学分子式是',
        '地球上最大的动物是',
        '地球自转一圈大约需要',
        '杭州市的美食有',
        '江苏省的最好的大学',
    ]

    # print(query(prompt_datas[0]))
    # sys.exit(0)

    qa_index = 0
    while True:
        start = time.time()
        if answer_way == 'user':
            # run generation
            prompt = input('用户：')
        else:
            if qa_index >= len(prompt_datas):
                break
            prompt = prompt_datas[qa_index]
            print('问题：', prompt)
            qa_index += 1

        prompt = tokenizer.bos_token + prompt
        x = tokenizer(prompt).data['input_ids']
        x = (torch.tensor(x, dtype=torch.long, device=device)[None, ...])

        res_y = model.generate(x, max_new_tokens=max_seq_len, temperature=temperature, do_sample=True)
        print('回答：', end='')
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(x, res_y)]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(response, end='', flush=True)
        print('\n')

        # with torch.no_grad():
        #     # res_y = model.generate(x, tokenizer.eos_token_id, max_new_tokens=max_seq_len, temperature=temperature,
        #     #                        top_k=top_k, stream=stream)

        #     try:
        #         y = next(res_y)
        #     except StopIteration:
        #         print("No answer")
        #         continue

        #     history_idx = 0
        #     while y != None:
        #         answer = tokenizer.decode(y[0].tolist())
        #         if answer and answer[-1] == '�':
        #             try:
        #                 y = next(res_y)
        #             except:
        #                 break
        #             continue
        #         # print(answer)
        #         if not len(answer):
        #             try:
        #                 y = next(res_y)
        #             except:
        #                 break
        #             continue

        #         print(answer[history_idx:], end='', flush=True)
        #         try:
        #             y = next(res_y)
        #         except:
        #             break
        #         history_idx = len(answer)
        #         if not stream:
        #             break

        #     print('\n')

        end = time.time()
        print(end - start, 's')
