



# pip install

Building wheels for collected packages: jieba, matplotlib, pandas, tiktoken, ujson
  Building wheel for jieba (setup.py) ... done
  Created wheel for jieba: filename=jieba-0.42.1-py3-none-any.whl size=19314459 sha256=310bddafc52c57473ff02ff5185c9064ac78bc3512e18a67928b51af082a83d8
  Stored in directory: /root/.cache/pip/wheels/1e/85/c2/4025913f4bed6d3cec95b76176354cdb1565930e081865d09b

  Building wheel for matplotlib (setup.py) ... -

      error: Failed to download any of the following: ['https://downloads.sourceforge.net/project/freetype/freetype2/2.6.1/freetype-2.6.1.tar.gz', 'https://download.savannah.gnu.org/releases/freetype/freetype-2.6.1.tar.gz', 'https://download.savannah.gnu.org/releases/freetype/freetype-old/freetype-2.6.1.tar.gz'].  Please download one of these urls and extract it into 'build/' at the top-level of the source repository.


Installing collected packages: sentencepiece, pytz, jieba, xxhash, ujson, tzdata, typeguard, threadpoolctl, smmap, simhash, shtab, setproctitle, sentry-sdk, scipy, safetensors, regex, pydantic-core, pyarrow-hotfix, pyarrow, propcache, ngrok, multidict, mdurl, marshmallow, jsonlines, joblib, jiter, jinja2, itsdangerous, fsspec, frozenlist, docstring-parser, docker-pycreds, dill, click, blinker, annotated-types, aiohappyeyeballs, yarl, tiktoken, scikit_learn, pydantic, pandas, nltk, multiprocess, markdown-it-py, gitdb, Flask, datasketch, aiosignal, tokenizers, rich, openai, gitpython, Flask_Cors, aiohttp, wandb, tyro, transformers, accelerate, sentence_transformers, peft, datasets, trl
  Attempting uninstall: jinja2
    Found existing installation: Jinja2 3.1.4
    Uninstalling Jinja2-3.1.4:
      Successfully uninstalled Jinja2-3.1.4
  Attempting uninstall: fsspec
    Found existing installation: fsspec 2024.5.0
    Uninstalling fsspec-2024.5.0:
      Successfully uninstalled fsspec-2024.5.0


Successfully installed Flask-3.0.3 Flask_Cors-4.0.0 accelerate-1.3.0 aiohappyeyeballs-2.4.4 aiohttp-3.11.11 aiosignal-1.3.2 annotated-types-0.7.0 blinker-1.9.0 click-8.1.8 datasets-2.16.1 datasketch-1.6.4 dill-0.3.7 docker-pycreds-0.4.0 docstring-parser-0.16 frozenlist-1.5.0 fsspec-2023.10.0 gitdb-4.0.12 gitpython-3.1.44 itsdangerous-2.2.0 jieba-0.42.1 jinja2-3.1.2 jiter-0.8.2 joblib-1.4.2 jsonlines-4.0.0 markdown-it-py-3.0.0 marshmallow-3.22.0 mdurl-0.1.2 multidict-6.1.0 multiprocess-0.70.15 ngrok-1.4.0 nltk-3.8 openai-1.42.0 pandas-2.2.3 peft-0.7.1 propcache-0.2.1 pyarrow-19.0.0 pyarrow-hotfix-0.6 pydantic-2.8.2 pydantic-core-2.20.1 pytz-2024.2 regex-2024.11.6 rich-13.7.1 safetensors-0.5.2 scikit_learn-1.5.1 scipy-1.15.1 sentence_transformers-2.3.1 sentencepiece-0.2.0 sentry-sdk-2.20.0 setproctitle-1.3.4 shtab-1.7.1 simhash-2.1.2 smmap-5.0.2 threadpoolctl-3.5.0 tiktoken-0.8.0 tokenizers-0.19.1 transformers-4.44.0 trl-0.11.3 typeguard-4.4.1 tyro-0.9.13 tzdata-2025.1 ujson-5.10.0 wandb-0.18.3 xxhash-3.5.0 yarl-1.18.3


datasets
	fsspec-2023.10.0
	removed fsspec-2024.5.0

jinja2-3.1.2
	removed Jinja2-3.1.4

# dataset

https://hf-mirror.com/

通过 hf-mirror 下载
```shell
$ export HF_ENDPOINT=https://hf-mirror.com
$ huggingface-cli download --repo-type dataset --local-dir . --resume-download jingyaogong/minimind_dataset pretrain_data.csv sft_data_single.csv
...
pretrain_data.csv:   6%|███████▌                                                                                                                          | 273M/4.66G [00:43<09:58, 7.32MB/s]
sft_data_single.csv:  30%|██████████████████████████████████████▌                                                                                         | 493M/1.63G [00:32<01:12, 15.7MB/s]
```

## details

https://www.modelscope.cn/datasets/deepctrl/deepctrl-sft-data/files
sft_data_zh.jsonl 17.19GB

`wget "https://www.modelscope.cn/datasets/deepctrl/deepctrl-sft-data/resolve/master/sft_data_zh.jsonl"`，太大了，可以暂时先不下载。直接使用分离好（且截断数据到 512）的数据 `sft_data_single.csv`。

同理，处理好的供预训练的数据 pretrain_data.csv。


https://huggingface.co/datasets/jingyaogong/minimind_dataset/tree/main

理论上仅需要两种 dataset
1-pretrain.py:    parser.add_argument("--data_path", type=str, default="./dataset/pretrain_data.csv", help="Path to training data")
1-pretrain.py:    df = pd.read_csv(args.data_path)

3-full_sft.py:    df = pd.read_csv('./dataset/sft_data_single.csv')
4-lora_sft.py:    df = pd.read_csv('./dataset/sft_data_single.csv')

数据处理
data_process.py:import jsonlines
data_process.py:    with jsonlines.open('./dataset/mobvoi_seq_monkey_general_open_corpus.jsonl') as reader:
data_process.py:    sft_datasets = ['./dataset/sft_data_zh.jsonl']
data_process.py:        sft_datasets = ['./dataset/sft_data_zh.jsonl']
data_process.py:        with jsonlines.open(path) as reader:
data_process.py:                except jsonlines.InvalidLineError as e:
train_tokenizer.py:    def read_texts_from_jsonl(file_path):
train_tokenizer.py:    data_path = './dataset/tokenizer_train.jsonl'

1. 预训练
./dataset/mobvoi_seq_monkey_general_open_corpus.jsonl -> ./dataset/pretrain_data.csv
2. QA 问答
./dataset/sft_data_zh.jsonl -> sft_data_single.csv, sft_data.csv
3. 强化学习（极少数据）
125     dataset_paths = [
126         './dataset/dpo/dpo_zh_demo.json',
127         './dataset/dpo/dpo_train_data.json',
128         './dataset/dpo/huozi_rlhf_data.json',
129     ]
	->
./dataset/dpo/train_data.json

0. 词表
(base) autodl-container-8d4e4393c8-f5a5d168 minimind # ls ./model/minimind_tokenizer -lhtr
total 416K
-rw-r--r-- 1 root root  94K Jan 27 23:40 vocab.json
-rw-r--r-- 1 root root 1.6K Jan 27 23:40 tokenizer_config.json
-rw-r--r-- 1 root root 256K Jan 27 23:40 tokenizer.json
-rw-r--r-- 1 root root  57K Jan 27 23:40 merges.txt

```shell
$ python data_process.py
The cache for model files in Transformers v4.22.0 has been updated. Migrating your old cache. This is a one-time only operation. You can interrupt this and resume the migration later on by calling `transformers.utils.move_cache()`.
0it [00:00, ?it/s]
tokenizer词表大小： 6400
```

# pretrain

## 环境测试
测试torch是否可用cuda

```
import torch
print(torch.cuda.is_available())
```

## 调整参数
`./model/LMConfig.py`
> dim和n_layers参数为(512+8)

`python 1-pretrain.py` 执行预训练，得到 `pretrain_*.pth` 作为预训练的输出权重

先尝试下无 GPU 卡执行，在 Load ./dataset/pretrain_data.csv 时被 OOM killed。
```
python 1-pretrain.py
2025-01-28 01:33:07,459 __main__ INFO Importing: from model.model import Transformer
2025-01-28 01:33:08,378 __main__ INFO Importing: from model.LMConfig import LMConfig
2025-01-28 01:33:08,378 __main__ INFO Importing: from model.dataset import PretrainDataset
2025-01-28 01:33:09,245 __main__ INFO Run with args: Namespace(out_dir='out', epochs=20, batch_size=64, learning_rate=0.0002, device='cpu', dtype='bfloat16', use_wandb=False, wandb_project='MiniMind-Pretrain', num_workers=1, data_path='./dataset/pretrain_data.csv', ddp=False, accumulation_steps=8, grad_clip=1.0, warmup_iters=0, log_interval=100, save_interval=1000, local_rank=-1)
2025-01-28 01:33:09,246 __main__ INFO init tokenizer
2025-01-28 01:33:09,252 __main__ INFO init model
LLM总参数量：26.878 百万
2025-01-28 01:33:10,054 __main__ INFO Read PretrainDataset csv ./dataset/pretrain_data.csv
Killed
```

2张 4090D

```
$ nvidia-smi
Tue Jan 28 02:05:18 2025
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.78                 Driver Version: 550.78         CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 4090 D      On  |   00000000:17:00.0 Off |                  Off |
| 68%   63C    P2            324W /  425W |   11573MiB /  24564MiB |    100%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   1  NVIDIA GeForce RTX 4090 D      On  |   00000000:98:00.0 Off |                  Off |
| 31%   52C    P2            303W /  425W |   11573MiB /  24564MiB |    100%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
+-----------------------------------------------------------------------------------------+
```

```
2025-01-28 02:02:25,933 __main__ INFO Importing: from model.model import Transformer
2025-01-28 02:02:26,243 __main__ INFO Importing: from model.LMConfig import LMConfig
2025-01-28 02:02:26,243 __main__ INFO Importing: from model.dataset import PretrainDataset
2025-01-28 02:02:26,290 __main__ INFO Importing: from model.model import Transformer
2025-01-28 02:02:26,670 __main__ INFO Run with args: Namespace(out_dir='out', epochs=20, batch_size=64, learning_rate=0.0002, device='cuda:0', dtype='bfloat16', use_wandb=False, wandb_project='MiniMind-Pretrain', num_workers=1, data_path='./dataset/pretrain_data.csv', ddp=False, accumulation_steps=8, grad_clip=1.0, warmup_iters=0, log_interval=100, save_interval=1000, local_rank=-1)
2025-01-28 02:02:26,767 __main__ INFO Importing: from model.LMConfig import LMConfig
2025-01-28 02:02:26,767 __main__ INFO Importing: from model.dataset import PretrainDataset
2025-01-28 02:02:26,819 __main__ INFO init tokenizer
2025-01-28 02:02:26,825 __main__ INFO init model
2025-01-28 02:02:27,258 __main__ INFO Read PretrainDataset csv ./dataset/pretrain_data.csv
2025-01-28 02:02:27,327 __main__ INFO Run with args: Namespace(out_dir='out', epochs=20, batch_size=64, learning_rate=0.0002, device='cuda:0', dtype='bfloat16', use_wandb=False, wandb_project='MiniMind-Pretrain', num_workers=1, data_path='./dataset/pretrain_data.csv', ddp=False, accumulation_steps=8, grad_clip=1.0, warmup_iters=0, log_interval=100, save_interval=1000, local_rank=-1)
2025-01-28 02:02:27,355 __main__ INFO init tokenizer
2025-01-28 02:02:27,364 __main__ INFO init model
LLM总参数量：26.878 百万
2025-01-28 02:02:27,901 __main__ INFO Read PretrainDataset csv ./dataset/pretrain_data.csv
2025-01-28 02:03:09,925 __main__ INFO Epoch:[0/20](0/41914) loss:8.879 lr:0.0002000 epoch_Time:763.0min:
2025-01-28 02:03:09,926 __main__ INFO Epoch:[0/20](0/41914) loss:8.875 lr:0.0002000 epoch_Time:849.0min:
Epoch:[0/20](0/41914) loss:8.875 lr:0.0002000 epoch_Time:849.0min:
2025-01-28 02:03:24,226 __main__ INFO Epoch:[0/20](100/41914) loss:7.435 lr:0.0002000 epoch_Time:106.0min:
2025-01-28 02:03:24,226 __main__ INFO Epoch:[0/20](100/41914) loss:7.442 lr:0.0002000 epoch_Time:106.0min:
Epoch:[0/20](100/41914) loss:7.435 lr:0.0002000 epoch_Time:106.0min:
2025-01-28 02:03:38,403 __main__ INFO Epoch:[0/20](200/41914) loss:6.916 lr:0.0002000 epoch_Time:102.0min:
2025-01-28 02:03:38,403 __main__ INFO Epoch:[0/20](200/41914) loss:6.912 lr:0.0002000 epoch_Time:102.0min:
Epoch:[0/20](200/41914) loss:6.916 lr:0.0002000 epoch_Time:102.0min:
2025-01-28 02:03:52,590 __main__ INFO Epoch:[0/20](300/41914) loss:6.491 lr:0.0002000 epoch_Time:101.0min:
Epoch:[0/20](300/41914) loss:6.491 lr:0.0002000 epoch_Time:101.0min:
2025-01-28 02:03:52,590 __main__ INFO Epoch:[0/20](300/41914) loss:6.504 lr:0.0002000 epoch_Time:101.0min:
2025-01-28 02:04:06,779 __main__ INFO Epoch:[0/20](400/41914) loss:6.083 lr:0.0002000 epoch_Time:101.0min:
2025-01-28 02:04:06,779 __main__ INFO Epoch:[0/20](400/41914) loss:6.148 lr:0.0002000 epoch_Time:101.0min:
Epoch:[0/20](400/41914) loss:6.083 lr:0.0002000 epoch_Time:101.0min:
2025-01-28 02:04:20,965 __main__ INFO Epoch:[0/20](500/41914) loss:5.718 lr:0.0002000 epoch_Time:99.0min:
2025-01-28 02:04:20,965 __main__ INFO Epoch:[0/20](500/41914) loss:5.682 lr:0.0002000 epoch_Time:99.0min:
Epoch:[0/20](500/41914) loss:5.682 lr:0.0002000 epoch_Time:99.0min:
...
```

checkpoint 产出

```
ls -lhtr out/pretrain_512.pth
-rw-r--r-- 1 root root 103M Jan 28 02:05 out/pretrain_512.pth
```

## examine checkpoint

```
$ strings out-bak/pretrain_512.pth | head
pretrain_512/data.pklFB
ZZZZZZZZZ
ccollections
OrderedDict
tok_embeddings.weightq
ctorch._utils
_rebuild_tensor_v2
storageq
ctorch
FloatStorage
```

```
# strings out-bak/pretrain_512.pth | grep pretrain_512 | sort
pretrain_512/.data/serialization_idFB)
pretrain_512/.data/serialization_idPK
pretrain_512/byteorderFB9
pretrain_512/byteorderPK
pretrain_512/data.pklFB
pretrain_512/data.pklPK
pretrain_512/data/0FB5
pretrain_512/data/0PK
pretrain_512/data/10FB:
pretrain_512/data/10PK
pretrain_512/data/11FB:
pretrain_512/data/11PK
pretrain_512/data/12FB:
...
```

```
# strings out-bak/pretrain_512.pth | grep pretrain_512 | sort | tail
pretrain_512/data/73FB:
pretrain_512/data/73PK
pretrain_512/data/7FB;
pretrain_512/data/7PK
pretrain_512/data/8FB;
pretrain_512/data/8PK
pretrain_512/data/9FB;
pretrain_512/data/9PK
pretrain_512/versionFB:
pretrain_512/versionPK
```

```
# strings out-bak/pretrain_512.pth | grep pretrain_512 | wc -l
156
```

## 从之前的权重文件中加载，继续训练

```
$ grep -r pretrain_ *.py
0-eval_pretrain.py:        ckp = f'./out/pretrain_{lm_config.dim}{moe_path}.pth'
1-pretrain.py:            ckp = f'{args.save_dir}/pretrain_{lm_config.dim}{moe_path}.pth'
1-pretrain.py:    parser.add_argument("--data_path", type=str, default="./dataset/pretrain_data.csv", help="Path to training data")
3-full_sft.py:        ckp = f'./out/pretrain_{lm_config.dim}{moe_path}.pth'
```

```
$ grep -r load_state_dict
0-eval_pretrain.py:        model.load_state_dict(state_dict, strict=False)
2-eval.py:        model.load_state_dict(state_dict, strict=False)
3-full_sft.py:        model.load_state_dict(state_dict, strict=False)
eval_ceval.py:        model.load_state_dict(state_dict, strict=False)
export_model.py:    lm_model.load_state_dict(state_dict, strict=False)
中文逐行注释/1-pretrain.py:    # model.load_state_dict(state_dict, strict=False)
```

```diff
diff --git a/1-pretrain.py b/1-pretrain.py
index d9b70e2..676f8ec 100644
--- a/1-pretrain.py
+++ b/1-pretrain.py
@@ -121,7 +121,22 @@ def init_model():

     logger.info('init model')
     model = Transformer(lm_config).to(args.device)
-    # moe_path = '_moe' if lm_config.use_moe else ''
+
+    # load checkpoint
+    moe_path = '_moe' if lm_config.use_moe else ''
+    ckp = f'{args.save_dir}/pretrain_{lm_config.dim}{moe_path}.pth'
+    state_dict = torch.load(ckp, map_location=args.device)
+    # modify dict key to strip out the '_orig_mod.' prefix?
+    unwanted_prefix = '_orig_mod.'
+    for k, v in list(state_dict.items()):
+        logger.info('Loading checkpoint, state_dict: k: %s, v: %s', k, v)
+        if k.startswith(unwanted_prefix):
+            logger.info('state_dict: k: %s, v: %s', k[len(unwanted_prefix):], v)
+            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
+        else:
+            logger.info('Skpping, state_dict: k: %s, v: %s', k, v)
+    logger.info('model.load_state_dict')
+    model.load_state_dict(state_dict, strict=False)

     Logger(f'LLM总参数量：{count_parameters(model) / 1e6:.3f} 百万')
     return model, tokenizer
```

## pretrain eval

预测下一词

```
minimind # grep 'OpenAI的价值' dataset/pretrain_data.csv
"两家公司没有披露具体款项，但《纽约时报》、彭博分别援引一位知情人士消息称，微软将向OpenAI投资100亿美元。美国财经媒体Semafor此前1月10日报道了这一数额。这笔资金还包括其它风险投资机构。包括新投
资在内，OpenAI的价值将达到290亿美元。
minimind # grep '价值是很高的' dataset/pretrain_data.csv
戴氏法器说焚香习俗在我国有着悠久的历史，古人焚香很多，所以香炉的用途很广，除了礼仪环境所需要用的熏衣外，还是书斋里便于诵阅、有益于理解及记忆的文玩清供。此外还有一种用途便是陵墓、寺庙及权势之
家烧香、拜佛、祭祖神之用。虽然香炉收藏相对小众一些，但是其收藏价值和历史价值不容忽视。从工艺造型来看，一些皇帝御赐的铜香炉由于其精的工艺而提升了其价值。因此，年代久远、雕工精美的铜香炉价值
不菲。另外，由于明清铜炉历史上经历诸多劫难，损失惨重，造成后仿炉和私款炉也已成为珍罕的历史文物。看来铜香炉的收藏价值是很高的，本厂还加工铜鼎，铜钟等各种青铜器欢迎您的订购。"
上面介绍的海参芡实米粥的做法，是不是大家已经都学会了呢，方法简单，而且营养价值是很高的哦，如果你在考虑吃什么，这道营养的粥品是不错的选择，促进排便防上火，营养多，养胃补血养心安神，是不错的明
目滋阴的好食材哦。"

# python 0-eval_pretrain.py
2025-01-28 14:02:38,200 __main__ INFO Imporing: from model.model import Transformer
2025-01-28 14:02:38,667 __main__ INFO Imporing: from model.LMConfig import LMConfig
2025-01-28 14:02:38,696 __main__ INFO Run with args: Namespace(out_dir='out', checkpoint='pretrain_512-202501280340.pth', from_transformers=False, auto=False, device='cuda:0', dtype='bfloat1
6')
2025-01-28 14:02:38,707 __main__ INFO Load from checkpoint out/pretrain_512-202501280340.pth
模型参数: 26.878464 百万 = 0.026878464 B (Billion)
用户：OpenAI的价值
回答：是很高的，而且还非常的有价值。
OpenAI的价值是非常高的，因为OpenAI的价值是很高的，它可以给消费者带来更多的利润，所以在市场上有很多的价值。比如说，OpenAI的价值是非常高，并且它有很好的价值，而且它的价值也非常的高，而且它的价
值也是非常高的。
OptenAI的价值是非常高的，它可以给消费者带来更多的利润，所以在市场上有很多的价值也是非常高的，而且它的价值也非常的高，它可以给消费者带来很多的利润，所以它不仅可以给消费者带来更多的利润，而且
它的价值也非常的高。
OptenAI的价值是非常高的，它可以给消费者带来更多的利润，而且它还具有非常好的经济效益，所以它的价值是非常高的，它可以帮助人们获得很多的经济效益，而且它的价值非常的高，它也非常的高昂，所以在市
场上也是非常受欢迎。
OptenAI的价值是非常高的，它可以给消费者带来很多的利润，而且它的价值也是非常高的，它可以给人们带来很多的经济效益，并且它的价值也非常高的，它可以给人们带来很多的利润，而且它的价值也非常的高，
它可以让人们的经济收入增加了很多，但是它也是非常不错的，它的价值也是非常高的，所以它的价值还是非常高的，它可以给人们带来很多的经济效益，而且它的价值也是非常的高的，它可以让人们的经济效益得到
非常的大大的进步，所以它的价值也是非常高的。

10.337987661361694 s
```

## from transformers

感觉效果更差？不太说人话

```
用户：OpenAI 是
回答：权权权权作为一个权权的，作为“权权权义、权作为政府的主权者、政府的主制力、作为权作为保障的，作为“公民权力的应力制作为。 （公义作为作为公义的将和民作为作为“公义、义、以、或或是以公民、自由、民、公义、以、公民、公民、公法、公民、公法、民法、公民、公民、民民、公义、民民和民民为为、公民、公民、公民、民民和民民的需要和保护，民为社会公民的保障，并为公民社会的责任和为、为为民民民民的、为民民民民为民民为和民民为和民民为为。
```

```
/root/.cache/huggingface
/root/.cache/huggingface/stored_tokens
/root/.cache/huggingface/token
/root/.cache/huggingface/hub
/root/.cache/huggingface/hub/datasets--jingyaogong--minimind_dataset
/root/.cache/huggingface/hub/datasets--jingyaogong--minimind_dataset/refs
/root/.cache/huggingface/hub/datasets--jingyaogong--minimind_dataset/refs/main
/root/.cache/huggingface/hub/version.txt
/root/.cache/huggingface/hub/models--jingyaogong--minimind-v1-small
/root/.cache/huggingface/hub/models--jingyaogong--minimind-v1-small/blobs
/root/.cache/huggingface/hub/models--jingyaogong--minimind-v1-small/blobs/0dcf00543768c1980ab25a3f3368af85e5a17cf5
/root/.cache/huggingface/hub/models--jingyaogong--minimind-v1-small/blobs/fa42a4fa3c7553ae4272e05e38256d9dd2cc8698
/root/.cache/huggingface/hub/models--jingyaogong--minimind-v1-small/blobs/76798b88e0de97ce894a28702406c47bc69940ff
/root/.cache/huggingface/hub/models--jingyaogong--minimind-v1-small/blobs/f6cde76909c47b8b3fd5582487c7d47cd63772a917afa30303f9a81cef1f5bd5
/root/.cache/huggingface/hub/models--jingyaogong--minimind-v1-small/blobs/23cfacabaaf1c84b8486293b306b9d43d63f3da8
/root/.cache/huggingface/hub/models--jingyaogong--minimind-v1-small/snapshots
/root/.cache/huggingface/hub/models--jingyaogong--minimind-v1-small/snapshots/fad3b10dd5e251cb6f8050e5a3c8080efbccfdd2
/root/.cache/huggingface/hub/models--jingyaogong--minimind-v1-small/snapshots/fad3b10dd5e251cb6f8050e5a3c8080efbccfdd2/config.json
/root/.cache/huggingface/hub/models--jingyaogong--minimind-v1-small/snapshots/fad3b10dd5e251cb6f8050e5a3c8080efbccfdd2/LMConfig.py
/root/.cache/huggingface/hub/models--jingyaogong--minimind-v1-small/snapshots/fad3b10dd5e251cb6f8050e5a3c8080efbccfdd2/model.py
/root/.cache/huggingface/hub/models--jingyaogong--minimind-v1-small/snapshots/fad3b10dd5e251cb6f8050e5a3c8080efbccfdd2/pytorch_model.bin
/root/.cache/huggingface/hub/models--jingyaogong--minimind-v1-small/snapshots/fad3b10dd5e251cb6f8050e5a3c8080efbccfdd2/generation_config.json
/root/.cache/huggingface/hub/models--jingyaogong--minimind-v1-small/refs
/root/.cache/huggingface/hub/models--jingyaogong--minimind-v1-small/refs/main
/root/.cache/huggingface/hub/models--jingyaogong--minimind-v1-small/.no_exist
/root/.cache/huggingface/hub/models--jingyaogong--minimind-v1-small/.no_exist/fad3b10dd5e251cb6f8050e5a3c8080efbccfdd2
/root/.cache/huggingface/hub/models--jingyaogong--minimind-v1-small/.no_exist/fad3b10dd5e251cb6f8050e5a3c8080efbccfdd2/adapter_config.json
/root/.cache/huggingface/hub/models--jingyaogong--minimind-v1-small/.no_exist/fad3b10dd5e251cb6f8050e5a3c8080efbccfdd2/model.safetensors
/root/.cache/huggingface/hub/models--jingyaogong--minimind-v1-small/.no_exist/fad3b10dd5e251cb6f8050e5a3c8080efbccfdd2/model.safetensors.index.json
/root/.cache/huggingface/hub/.locks
/root/.cache/huggingface/hub/.locks/models--jingyaogong--minimind-v1-small
/root/.cache/huggingface/hub/.locks/models--jingyaogong--minimind-v1-small/0dcf00543768c1980ab25a3f3368af85e5a17cf5.lock
/root/.cache/huggingface/hub/.locks/models--jingyaogong--minimind-v1-small/fa42a4fa3c7553ae4272e05e38256d9dd2cc8698.lock
/root/.cache/huggingface/hub/.locks/models--jingyaogong--minimind-v1-small/76798b88e0de97ce894a28702406c47bc69940ff.lock
/root/.cache/huggingface/hub/.locks/models--jingyaogong--minimind-v1-small/f6cde76909c47b8b3fd5582487c7d47cd63772a917afa30303f9a81cef1f5bd5.lock
/root/.cache/huggingface/hub/.locks/models--jingyaogong--minimind-v1-small/23cfacabaaf1c84b8486293b306b9d43d63f3da8.lock
/root/.cache/huggingface/modules
/root/.cache/huggingface/modules/__init__.py
/root/.cache/huggingface/modules/transformers_modules
/root/.cache/huggingface/modules/transformers_modules/__init__.py
/root/.cache/huggingface/modules/transformers_modules/jingyaogong
/root/.cache/huggingface/modules/transformers_modules/jingyaogong/__init__.py
/root/.cache/huggingface/modules/transformers_modules/jingyaogong/minimind-v1-small
/root/.cache/huggingface/modules/transformers_modules/jingyaogong/minimind-v1-small/__init__.py
/root/.cache/huggingface/modules/transformers_modules/jingyaogong/minimind-v1-small/fad3b10dd5e251cb6f8050e5a3c8080efbccfdd2
/root/.cache/huggingface/modules/transformers_modules/jingyaogong/minimind-v1-small/fad3b10dd5e251cb6f8050e5a3c8080efbccfdd2/__init__.py
/root/.cache/huggingface/modules/transformers_modules/jingyaogong/minimind-v1-small/fad3b10dd5e251cb6f8050e5a3c8080efbccfdd2/LMConfig.py
/root/.cache/huggingface/modules/transformers_modules/jingyaogong/minimind-v1-small/fad3b10dd5e251cb6f8050e5a3c8080efbccfdd2/__pycache__
/root/.cache/huggingface/modules/transformers_modules/jingyaogong/minimind-v1-small/fad3b10dd5e251cb6f8050e5a3c8080efbccfdd2/__pycache__/LMConfig.cpython-312.pyc
/root/.cache/huggingface/modules/transformers_modules/jingyaogong/minimind-v1-small/fad3b10dd5e251cb6f8050e5a3c8080efbccfdd2/__pycache__/model.cpython-312.pyc
/root/.cache/huggingface/modules/transformers_modules/jingyaogong/minimind-v1-small/fad3b10dd5e251cb6f8050e5a3c8080efbccfdd2/model.py
```

## 最终训练 3 个 epoch

```
2025-01-28 16:29:46,792 __main__ INFO Epoch:[1/20](41600/41914) loss:2.406 lr:0.0001956 epoch_Time:0.0min:
2025-01-28 16:30:00,924 __main__ INFO Epoch:[1/20](41700/41914) loss:2.578 lr:0.0001956 epoch_Time:0.0min:
2025-01-28 16:30:00,924 __main__ INFO Epoch:[1/20](41700/41914) loss:2.496 lr:0.0001956 epoch_Time:0.0min:
Epoch:[1/20](41700/41914) loss:2.578 lr:0.0001956 epoch_Time:0.0min:
2025-01-28 16:30:15,059 __main__ INFO Epoch:[1/20](41800/41914) loss:2.431 lr:0.0001956 epoch_Time:0.0min:
Epoch:[1/20](41800/41914) loss:2.431 lr:0.0001956 epoch_Time:0.0min:
2025-01-28 16:30:15,059 __main__ INFO Epoch:[1/20](41800/41914) loss:2.475 lr:0.0001956 epoch_Time:0.0min:
2025-01-28 16:30:29,192 __main__ INFO Epoch:[1/20](41900/41914) loss:2.425 lr:0.0001956 epoch_Time:0.0min:
Epoch:[1/20](41900/41914) loss:2.425 lr:0.0001956 epoch_Time:0.0min:
2025-01-28 16:30:29,193 __main__ INFO Epoch:[1/20](41900/41914) loss:2.597 lr:0.0001956 epoch_Time:0.0min:

2025-01-28 16:30:32,595 __main__ INFO Epoch:[2/20](0/41914) loss:2.641 lr:0.0001956 epoch_Time:594.0min:
2025-01-28 16:30:32,595 __main__ INFO Epoch:[2/20](0/41914) loss:2.450 lr:0.0001956 epoch_Time:497.0min:
Epoch:[2/20](0/41914) loss:2.450 lr:0.0001956 epoch_Time:497.0min:
2025-01-28 16:30:46,743 __main__ INFO Epoch:[2/20](100/41914) loss:2.461 lr:0.0001956 epoch_Time:103.0min:
2025-01-28 16:30:46,743 __main__ INFO Epoch:[2/20](100/41914) loss:2.422 lr:0.0001956 epoch_Time:104.0min:
Epoch:[2/20](100/41914) loss:2.422 lr:0.0001956 epoch_Time:104.0min:
2025-01-28 16:31:00,887 __main__ INFO Epoch:[2/20](200/41914) loss:2.478 lr:0.0001956 epoch_Time:101.0min:
2025-01-28 16:31:00,887 __main__ INFO Epoch:[2/20](200/41914) loss:2.512 lr:0.0001956 epoch_Time:101.0min:
Epoch:[2/20](200/41914) loss:2.512 lr:0.0001956 epoch_Time:101.0min:
2025-01-28 16:31:15,061 __main__ INFO Epoch:[2/20](300/41914) loss:2.417 lr:0.0001956 epoch_Time:100.0min:
2025-01-28 16:31:15,062 __main__ INFO Epoch:[2/20](300/41914) loss:2.600 lr:0.0001956 epoch_Time:100.0min:
Epoch:[2/20](300/41914) loss:2.600 lr:0.0001956 epoch_Time:100.0min:
```

# wandb

wandb: Using wandb-core as the SDK backend. Please refer to https://wandb.me/wandb-core for more information.
wandb: Currently logged in as: tclh123 (tclh123ai). Use `wandb login --relogin` to force relogin
wandb: Tracking run with wandb version 0.18.3
wandb: Run data is saved locally in /root/projects/minimind/wandb/run-20250128_170218-g6gfzutl
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run MiniMind-Full-SFT-Epoch-19-BatchSize-32-LearningRate-1e-05
wandb: ⭐️ View project at https://wandb.ai/tclh123ai/MiniMind-Full-SFT
wandb: 🚀 View run at https://wandb.ai/tclh123ai/MiniMind-Full-SFT/runs/g6gfzutl

# sft

learning_rate 1e-5

```
LLM总参数量：26.878 百万
Epoch:[0/19](0/61701) loss:3.074 lr:0.0000100 epoch_Time:1479.0min:
Epoch:[0/19](100/61701) loss:2.683 lr:0.0000100 epoch_Time:87.0min:
Epoch:[0/19](200/61701) loss:2.478 lr:0.0000100 epoch_Time:81.0min:
Epoch:[0/19](300/61701) loss:2.589 lr:0.0000100 epoch_Time:79.0min:
Epoch:[0/19](400/61701) loss:2.463 lr:0.0000100 epoch_Time:78.0min:
Epoch:[0/19](500/61701) loss:2.461 lr:0.0000100 epoch_Time:77.0min:
Epoch:[0/19](600/61701) loss:2.354 lr:0.0000100 epoch_Time:76.0min:
Epoch:[0/19](700/61701) loss:2.446 lr:0.0000100 epoch_Time:76.0min:
Epoch:[0/19](800/61701) loss:2.442 lr:0.0000100 epoch_Time:75.0min:
```

产物
full_sft_512.pth

# 总体步骤

> 2.4 python 1-pretrain.py 执行预训练，得到 pretrain_*.pth 作为预训练的输出权重
> 2.5 python 3-full_sft.py 执行指令微调，得到 full_sft_*.pth 作为指令微调的输出权重
> 2.6 python 4-lora_sft.py 执行lora微调（非必须）
> 2.7 python 5-dpo_train.py 执行DPO人类偏好强化学习对齐（非必须）


# Others

see also https://github.com/jingyaogong/minimind/issues/26#issuecomment-2362938042

# Knowledge

https://huggingface.co/docs/transformers/main/chat_templating

https://github.com/jingyaogong/minimind/wiki

Quick Links

Read model documentation
1. https://huggingface.co/docs/transformers/main/en/model_doc/llama#transformers.LlamaForCausalLM
2. https://huggingface.co/docs/transformers/main/en/model_doc/qwen2#transformers.Qwen2ForCausalLM

Read docs on high-level-pipeline
https://huggingface.co/docs/transformers/main_classes/pipelines
Read our learning resources
https://huggingface.co/learn

