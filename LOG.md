



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
