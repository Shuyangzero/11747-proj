import argparse


def get_train_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name",
                        type=str,
                        default=None,
                        choices=["sentihood_NLI_M", "sentihood_NLI_B", "sentihood_single",
                                 "sentihood_QA_B", "sentihood_QA_M"])
    parser.add_argument("--data_dir",
                        type=str,
                        default='data',
                        required=True)
    parser.add_argument("--vocab_file",
                        type=str,
                        default='vocab',
                        required=True)
    parser.add_argument("--bert_config_file",
                        type=str,
                        default="",
                        required=True)
    parser.add_argument("--init_checkpoint",
                        type=str,
                        default=None)
    parser.add_argument("--eval_test",
                        action='store_true',
                        default=False)
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        default=False)
    parser.add_argument("--max_seq_length",
                        type=int,
                        default=256)
    parser.add_argument("--train_batch_size",
                        type=int,
                        default=16)
    parser.add_argument("--learning_rate",
                        type=float,
                        default=1e-4)
    parser.add_argument("--num_train_epochs",
                        type=float,
                        default=2.0)
    parser.add_argument("--output_dir",
                        type=str,
                        default="output/",
                        required=True)
    parser.add_argument('--seed',
                        type=int,
                        default=122)

    # Other parameters, delete help later
    parser.add_argument("--eval_batch_size",
                        type=int,
                        default=8,
                        help="Total batch size for eval.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                        "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--accumulate_gradients",
                        type=int,
                        default=1,
                        help="Number of steps to accumulate gradient on (divide the batch_size and accumulate)")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")

    args = parser.parse_args()
    return args


def get_eval_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        choices=["sentihood_single", "sentihood_NLI_M", "sentihood_QA_M",
                                 "sentihood_NLI_B", "sentihood_QA_B", "semeval_single",
                                 "semeval_NLI_M", "semeval_QA_M", "semeval_NLI_B", "semeval_QA_B"],
                        help="The name of the task to evalution.")
    parser.add_argument("--pred_data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The pred data dir.")
    args = parser.parse_args()
    return args
