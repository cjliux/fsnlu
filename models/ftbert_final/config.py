#coding: utf-8
"""
    @author: cjliux@gmail.com
"""

def add_parser_args(parser):
    parser.add_argument("--exp_name", type=str, default="ftbert_final", help="Experiment name")
    parser.add_argument("--exp_id", type=str, default="default", help="Experiment id")
    parser.add_argument("--log_file", type=str, default="fs_slu.log")
    parser.add_argument("--dump_path", type=str, default="ftbert_final_exp", help="Experiment saved root path")
    parser.add_argument("--target", type=str, default="best_model_{}.pth")
    parser.add_argument("--no_pbar", action='store_true')
    parser.add_argument("--save_dir", type=str, help="save path of prediction")
    parser.add_argument("--model_path", type=str, help="path to model to be loaded")

    parser.add_argument("--bert_dir", type=str, default="../resource/baidu_ernie")
    parser.add_argument("--do_lower_case", action='store_true')
    parser.add_argument("--max_seq_length", type=int, default=100)
    # parser.add_argument("--emb_dim", type=int, default=400, help="embedding dimension")
    parser.add_argument("--dropout", type=float, default=0.3, help="dropout rate")

    parser.add_argument("--data_path", type=str, 
                        default="./data/default_v4")
    parser.add_argument("--raw_data_path", type=str, 
                        default="../data/smp2020ecdt/smp2020ecdt_task1_v2")
    parser.add_argument("--fin_data_path", type=str,
                        default="../data/smp2020ecdt/test_support_v4")
    parser.add_argument("--evl_dm", type=str, help="eval_domains", 
                        default="")
    parser.add_argument("--tst_dm", type=str, help="test_domains", 
                        default="all")
    # parser.add_argument("--tst_dm", type=str, help="target_domain")
    parser.add_argument("--load_userdict", action='store_true')
    parser.add_argument("--userdict", type=str, 
                        default="./data/dictionary.txt")

    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
    parser.add_argument("--max_sup_size", type=int, default=28)
    parser.add_argument("--max_sup_ratio", type=float, default=0.2)
    parser.add_argument("--n_shots", type=int, default=5, help="num shots")
    parser.add_argument("--max_epoch", type=int, default=12, help="number of maximum epoch")
    parser.add_argument("--lr", type=float, default=6e-5, help="learning rate")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--early_stop", type=int, default=0, help="No improvement after several epoch, we stop training")
    # parser.add_argument('--grad_acc_steps', type=int, default=1,
    #                     help="Number of updates steps to accumulate before performing a backward/update pass.")

    return parser
