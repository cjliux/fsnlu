#coding: utf-8

def add_parser_args(parser):
    parser.add_argument("--exp_name", type=str, default="default", help="Experiment name")
    parser.add_argument("--log_file", type=str, default="fs_slu.log")
    parser.add_argument("--dump_path", type=str, default="fsbert_exp", help="Experiment saved root path")
    parser.add_argument("--exp_id", type=str, default="1", help="Experiment id")
    
    parser.add_argument("--bert_dir", type=str, 
        default="../resource/baidu_ernie")
    parser.add_argument("--do_lower_case", action='store_true')
    # parser.add_argument("--emb_dim", type=int, default=400, help="embedding dimension")
    parser.add_argument("--dropout", type=float, default=0.3, help="dropout rate")

    parser.add_argument("--data_path", type=str, default="./data/default")
    parser.add_argument("--raw_data_path", type=str, 
        default="../data/smp2020ecdt/smp2020ecdt_task1_v2")
    parser.add_argument("--evl_dm", type=str, help="eval_domains",
        default="cookbook,website")

    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--n_shots", type=int, default=7, help="num shots")
    parser.add_argument("--max_epoch", type=int, default=3, help="number of maximum epoch")
    parser.add_argument("--lr", type=float, default=5e-5, help="learning rate")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--early_stop", type=int, default=5, help="No improvement after several epoch, we stop training")
    parser.add_argument('--grad_acc_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")

    return parser
