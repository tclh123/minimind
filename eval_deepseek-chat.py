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

# logger.info('Imporing: from model.model import Transformer')
# from model.model import Transformer
# logger.info('Imporing: from model.LMConfig import LMConfig')
# from model.LMConfig import LMConfig

warnings.filterwarnings('ignore')

# "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
# 大概需要使用 11GiB 显存


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def init_model():
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepSeek eval chat")
    # parser.add_argument("--out_dir", type=str, default="out", help="Output directory")
    # parser.add_argument("--checkpoint", '-p', type=str, default="pretrain_512-202501280340.pth", help="the checkpoint filename to load")
    # parser.add_argument("--from-transformers", '-t', action='store_true', default=False, help="whether to load from transformers")
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
    # out_dir = args.out_dir
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
    # lm_config = LMConfig()
    # lm_config.max_seq_len = max_seq_len
    # -----------------------------------------------------------------------------

    model, tokenizer = init_model()
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

    qa_index = 0
    while True:
        if answer_way == 'user':
            # run generation
            prompt = input('用户：')
        else:
            if qa_index >= len(prompt_datas):
                break
            prompt = prompt_datas[qa_index]
            print('问题：', prompt)
            qa_index += 1

        start = time.time()

        # prompt = tokenizer.bos_token + prompt

        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        print('rendering messages: ', text)
        model_inputs = tokenizer([text], return_tensors="pt").to(device)

        x = model_inputs.input_ids
        # x = tokenizer(prompt).data['input_ids']
        # x = (torch.tensor(x, dtype=torch.long, device=device)[None, ...])

        res_y = model.generate(x, max_new_tokens=max_seq_len, temperature=temperature, do_sample=True)
        print('回答：', end='')
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(x, res_y)]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(response, end='', flush=True)
        print('\n')

        end = time.time()
        print(end - start, 's')
