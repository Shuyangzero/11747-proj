# ABSA as a Sentence Pair Classification Task

Codes for paper "Utilizing BERT for Aspect-Based Sentiment Analysis via Constructing Auxiliary Sentence" (NAACL 2019)

## Requirement

* pytorch: 1.0.0
* python: 3.7.1
* tensorflow: 1.13.1 (only needed for converting BERT-tensorflow-model to pytorch-model)
* numpy: 1.15.4
* nltk
* sklearn

## Step 1: prepare datasets

### SentiHood

Run following commands to prepare sentihodd dataset for tasks:

```
cd generate/
bash make.sh sentihood
```

## Step 2: prepare BERT-pytorch-model

Download [BERT-Base (Google's pre-trained models)](https://github.com/google-research/bert) and then convert a tensorflow checkpoint to a pytorch model.

Run:

```
python convert_tf_checkpoint_to_pytorch.py \
--tf_checkpoint_path uncased_L-12_H-768_A-12/bert_model.ckpt \
--bert_config_file uncased_L-12_H-768_A-12/bert_config.json \
--pytorch_dump_path uncased_L-12_H-768_A-12/pytorch_model.bin
```

## Step 3: train

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_classifier_TABSA.py \
--task_name sentihood_QA_M \
--data_dir data/sentihood/bert-pair/ \
--vocab_file uncased_L-12_H-768_A-12/vocab.txt \
--bert_config_file uncased_L-12_H-768_A-12/bert_config.json \
--init_checkpoint uncased_L-12_H-768_A-12/pytorch_model.bin \
--eval_test \
--do_lower_case \
--max_seq_length 512 \
--train_batch_size 24 \
--learning_rate 2e-5 \
--num_train_epochs 4 \
--output_dir results/sentihood/QA_M \
--seed 42
```

## Step 4: evaluation

Evaluate the results on test set (calculate Acc, F1, etc.):

```
python evaluation.py --task_name sentihood_QA_M --pred_data_dir results/sentihood/QA_M/test_ep_4.txt
```