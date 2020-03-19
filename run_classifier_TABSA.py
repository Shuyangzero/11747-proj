# coding=utf-8

"""BERT finetuning runner."""

from __future__ import absolute_import, division, print_function
from arguments import get_train_arguments
import argparse
import collections
import logging
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from tqdm import tqdm, trange

# import tokenization
from tokenization import FullTokenizer
from modeling import BertConfig, BertForSequenceClassification
from optimization import BERTAdam
from processor import (Semeval_NLI_B_Processor, Semeval_NLI_M_Processor,
                       Semeval_QA_B_Processor, Semeval_QA_M_Processor,
                       Semeval_single_Processor, Sentihood_NLI_B_Processor,
                       Sentihood_NLI_M_Processor, Sentihood_QA_B_Processor,
                       Sentihood_QA_M_Processor, Sentihood_single_Processor)


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


def get_features(egs, labels, max_sequence_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    count = 0
    lmap = {}
    for label in labels:
        lmap[label] = count
        count += 1

    features = []
    for eg in tqdm(egs):
        tokB = None
        tok_A = tokenizer.tokenize(eg.text_a)

        if eg.text_b:
            tokB = tokenizer.tokenize(eg.text_b)

        if tokB:

            while len(tok_A) + len(tokB) > max_sequence_length - 3:
                if len(tok_A) <= len(tokB):
                    tokB.pop()
                else:
                    tok_A.pop()

        else:
            # Account for [CLS] and [SEP] with "- 2"
            maxl = max_sequence_length - 2
            tok_A = tok_A[0:maxl] if len(tok_A) > maxl else tok_A

        # two types of token conversion
        tokens = [i for i in tok_A]
        tokens.append("[SEP]")
        tokens.insert(0, "[CLS]")
        seg_idxs = [0 for i in tokens]

        if tokB:
            for i, t in enumerate(tokB):
                tokens.append(t)
                seg_idxs.append(1)
            tokens.append("[SEP]")
            seg_idxs.append(1)

        f = get_f(tokens, tokenizer, max_sequence_length,
                  seg_idxs, lmap, eg)
        features.append(f)
    return features


def get_f(tokens, tokenizer, max_sequence_length, seg_idxs, lmap, eg):
    # convert tokens into feature
    in_idxs = tokenizer.convert_tokens_to_ids(tokens)
    pad_length = len(in_idxs)
    while len(seg_idxs) < max_sequence_length:
        seg_idxs.append(0)
        in_idxs.append(0)
    zero_length = len(in_idxs) - pad_length
    mask = pad_length * [1] + zero_length * [0]
    f = InputFeatures(input_ids=in_idxs, input_mask=mask,
                      segment_ids=seg_idxs, label_id=lmap[eg.label])
    return f


def get_input_tensor(features):
    l_ids = torch.tensor(
        [f.label_id for f in features], dtype=torch.long)
    i_ids = torch.tensor(
        [f.input_ids for f in features], dtype=torch.long)
    s_ids = torch.tensor(
        [f.segment_ids for f in features], dtype=torch.long)
    i_mks = torch.tensor(
        [f.input_mask for f in features], dtype=torch.long)
    return l_ids, i_ids, s_ids, i_mks


def main():
    args = get_train_arguments()
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    num_gpu = torch.cuda.device_count()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if num_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    bert_config = BertConfig.from_json_file(args.bert_config_file)
    tokenizer = FullTokenizer(
        vocab_file=args.vocab_file, do_lower_case=args.do_lower_case)

    processor = None
    if args.task_name == "sentihood_NLI_M":
        processor = Sentihood_NLI_M_Processor()
    elif args.task_name == "sentihood_NLI_B":
        processor = Sentihood_NLI_B_Processor
    elif args.task_name == "sentihood_single":
        processor = Sentihood_single_Processor()
    elif args.task_name == "sentihood_QA_B":
        processor = Sentihood_QA_B_Processor()
    elif args.task_name == "sentihood_QA_M":
        processor = Sentihood_QA_M_Processor()
    else:
        raise ValueError("Unimplemented task!")

    if not os.path.exists(args.output_dir):
        print('make output directory {}'.format(args.output_dir))
        os.makedirs(args.output_dir)

    labels = processor.get_labels()

    # training set
    if not os.path.exists(args.data_dir):
        raise ValueError("Data does not exist")
    train_examples = processor.get_train_examples(args.data_dir)

    divide = args.train_batch_size * args.num_train_epochs
    num_train_steps = int(len(train_examples) / divide)

    train_features = get_features(
        train_examples, labels, args.max_seq_length, tokenizer)

    label_ids, input_ids, seg_ids, input_mks = get_input_tensor(
        train_features)

    train_dataset = TensorDataset(input_ids, input_mks, seg_ids, label_ids)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    # test set
    if args.eval_test:
        test_examples = processor.get_test_examples(args.data_dir)
        test_features = get_features(
            test_examples, labels, args.max_seq_length, tokenizer)
        label_ids, input_ids, seg_ids, input_mks = get_input_tensor(
            test_features)
        test_data = TensorDataset(
            input_ids, input_mks, seg_ids, label_ids)
        test_dataloader = DataLoader(
            test_data, batch_size=args.eval_batch_size, shuffle=False)

    # model and optimizer
    model = BertForSequenceClassification(bert_config, len(labels))
    # load the pretrained parameters
    model.bert.load_state_dict(torch.load(
        args.init_checkpoint, map_location='cpu'))
    model.to(device)
    if num_gpu > 1:
        model = torch.nn.DataParallel(model)

    ##########################continue here#########################################
    with open('{}/log.txt'.format(args.output_dir), "w") as f:
        title = "epoch global_step loss\ttest_loss test_accuracy" if args.eval_test else "epoch global_step loss "
        f.write(title)
        f.write("\n")

    no_decay = ['bias', 'gamma', 'beta']
    optimizer_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]

    # train

    optimizer = BERTAdam(optimizer_parameters,
                         lr=args.learning_rate,
                         warmup=args.warmup_proportion,
                         t_total=num_train_steps)

    global_step = 0
    epoch = 0
    for ti in trange(int(args.num_train_epochs), desc="Epoch"):
        nb_tr_examples, nb_tr_steps, tr_loss = 0, 0, 0
        epoch += 1
        model.train()

        for i, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            # batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            input_ids, input_mask, segment_ids, label_ids = input_ids.to(
                device), input_mask.to(device), segment_ids.to(device), label_ids.to(device)
            loss, _ = model(input_ids, segment_ids, input_mask, label_ids)
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            if num_gpu > 1:
                loss = loss.mean()
            loss.backward()
            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            optimizer.step()
            model.zero_grad()
            global_step += 1

        # eval_test
        if args.eval_test:
            model.eval()
            test_loss, test_accuracy, nb_test_steps, nb_test_examples = 0, 0, 0, 0
            fname = "{}/test_ep_{}.txt".format(args.output_dir, epoch)
            with open(fname, "w") as ftname:
                for batch in test_dataloader:
                    input_ids, input_mask, segment_ids, label_ids = batch
                    input_ids, input_mask = input_ids.to(
                        device), input_mask.to(device),
                    segment_ids, label_ids = segment_ids.to(
                        device), label_ids.to(device)
                    with torch.no_grad():
                        tmp_test_loss, logits = model(
                            input_ids, segment_ids, input_mask, label_ids)

                    label_ids = label_ids.to('cpu').numpy()
                    logits = F.softmax(logits, dim=-1).detach().cpu().numpy()
                    outputs = np.argmax(logits, axis=1)
                    for o_i in range(len(outputs)):
                        ftname.write(str(outputs[o_i]))
                        for ou in logits[o_i]:
                            ftname.write(" " + str(ou))
                        ftname.write("\n")

                    test_accuracy += np.sum(outputs == label_ids)
                    test_loss += tmp_test_loss.mean().item()
                    nb_test_examples += input_ids.size(0)
                    nb_test_steps += 1

            test_loss = test_loss / nb_test_steps
            test_accuracy = test_accuracy / nb_test_examples

        eval_str = "{} {} {} {}\n".format(epoch, global_step, tr_loss /
                                          nb_tr_steps, test_loss, test_accuracy)
        train_str = "{} {} {}\n".format(
            epoch, global_step, tr_loss / nb_tr_steps)
        row = eval_str if args.eval_test else train_str
        with open('{}/log.txt'.format(args.output_dir), "a+") as f:
            f.write(row)


if __name__ == "__main__":
    main()
