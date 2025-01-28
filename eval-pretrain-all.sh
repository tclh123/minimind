#!/bin/bash


# python 0-eval_pretrain.py --checkpoint pretrain_512-202501280340.pth --auto &> logs/0-eval-pretrain_512-202501281628.pth.log &

for f in out/pretrain_512-20250128*; do
	echo $f
	filename=${f#*/}
	echo python 0-eval_pretrain.py --checkpoint "$filename" --auto
	python 0-eval_pretrain.py --checkpoint "$filename" --auto &> "logs/0-eval-$filename.log"
done
