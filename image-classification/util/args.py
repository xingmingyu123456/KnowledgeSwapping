import argparse


def get_args():
    parser = argparse.ArgumentParser(description="for knowledge swapping")
    parser.add_argument("--learnpath",default="/home/xmy/code/classify-bird/dataset/classification10/crop_disease_learn",type=str)
    parser.add_argument("--forgetpath",default="/home/xmy/code/classify-bird/dataset/classification10/imagenet100_small_forget",type=str)
    parser.add_argument("--retainpath",default="/home/xmy/code/classify-bird/dataset/classification10/imagenet100_small_remain",type=str)


    parser.add_argument(
        "-w", "--workers_id", help="gpu ids or cpu", default="cpu", type=str
    )
    parser.add_argument("-e", "--epochs", help="training epochs", default=125, type=int)
    parser.add_argument("-b", "--batch_size", help="batch_size", default=256, type=int)
    parser.add_argument(
        "-d",
        "--data_mode",
        help="use which database, [casia100, casia1000]",
        default="casia100",
        type=str,
    )
    parser.add_argument(
        "-n", "--net", help="which network, ['VIT','VITs']", default="VITs", type=str
    )
    parser.add_argument(
        "-head",
        "--head",
        help="head type, ['Softmax', 'ArcFace', 'CosFace', 'SFaceLoss']",
        default="ArcFace",
        type=str,
    )
    parser.add_argument("-r", "--resume", help="resume model", default="", type=str)
    parser.add_argument("--outdir", help="output dir", default="", type=str)

    parser.add_argument(
        "--opt",
        default="adamw",
        type=str,
        metavar="OPTIMIZER",
        help='Optimizer (default: "adamw"',
    )
    parser.add_argument(
        "--opt-eps",
        default=1e-8,
        type=float,
        metavar="EPSILON",
        help="Optimizer Epsilon (default: 1e-8)",
    )
    parser.add_argument(
        "--opt-betas",
        default=None,
        type=float,
        nargs="+",
        metavar="BETA",
        help="Optimizer Betas (default: None, use opt default)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        metavar="M",
        help="SGD momentum (default: 0.9)",
    )
    parser.add_argument(
        "--weight-decay", type=float, default=0.05, help="weight decay (default: 0.05)"
    )
    # Learning rate schedule parameters
    parser.add_argument(
        "--sched",
        default="cosine",
        type=str,
        metavar="SCHEDULER",
        help='LR scheduler (default: "cosine"',
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-4,
        metavar="LR",
        help="learning rate (default: 5e-4)",
    )
    parser.add_argument(
        "--lr-noise",
        type=float,
        nargs="+",
        default=None,
        metavar="pct, pct",
        help="learning rate noise on/off epoch percentages",
    )
    parser.add_argument(
        "--lr-noise-pct",
        type=float,
        default=0.67,
        metavar="PERCENT",
        help="learning rate noise limit percent (default: 0.67)",
    )
    parser.add_argument(
        "--lr-noise-std",
        type=float,
        default=1.0,
        metavar="STDDEV",
        help="learning rate noise std-dev (default: 1.0)",
    )
    parser.add_argument(
        "--warmup-lr",
        type=float,
        default=1e-6,
        metavar="LR",
        help="warmup learning rate (default: 1e-6)",
    )
    parser.add_argument(
        "--min-lr",
        type=float,
        default=1e-5,
        metavar="LR",
        help="lower lr bound for cyclic schedulers that hit 0 (1e-5)",
    )

    parser.add_argument(
        "--decay-epochs",
        type=int,
        default=30,
        metavar="N",
        help="epoch interval to decay LR",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=3,
        metavar="N",
        help="epochs to warmup LR, if scheduler supports",
    )
    parser.add_argument(
        "--cooldown-epochs",
        type=int,
        default=10,
        metavar="N",
        help="epochs to cooldown LR at min_lr, after cyclic schedule ends",
    )
    parser.add_argument(
        "--patience-epochs",
        type=int,
        default=10,
        metavar="N",
        help="patience epochs for Plateau LR scheduler (default: 10",
    )
    parser.add_argument(
        "--decay-rate",
        "--dr",
        type=float,
        default=0.1,
        metavar="RATE",
        help="LR decay rate (default: 0.1)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        metavar="N",
        help="dataloader threads (default: 4)",
    )

    # lora rank on FFN of Transformer blocks
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=8,
        metavar="N",
        help="lora rank on FFN of Transformer blocks (default: 8)",
    )
    # lora pos (FFN and attention) on Transformer blocks
    parser.add_argument(
        "--lora_pos",
        type=str,
        default="FFN",
        help="lora pos (FFN and attention) on Transformer blocks (default: FFN)",
    )
    # wandb offline
    parser.add_argument(
        "--wandb_offline",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--wandb_group", default=None, type=str, help="wandb group name"
    )
    # VIT depth
    parser.add_argument(
        "--vit_depth", type=int, default=12, metavar="N", help="vit depth (default: 6)"
    )

    # add forget parameters
    parser.add_argument(
        "--num_of_first_cls", type=int, default=90, help="number of first class"
    )
    parser.add_argument("--per_forget_cls", type=int, default=10)
    parser.add_argument("--BND", type=float, default=10)
    parser.add_argument("--beta", type=float, default=0.03)
    parser.add_argument("--alpha", type=float, default=0.1)

    # mode selection
    parser.add_argument(
        "--one_stage",
        default=True,
        action="store_false",
        help="whether to use one stage training",
    )
    parser.add_argument(
        "--l2", default=False, action="store_true", help="whether to use l2 norm"
    )
    parser.add_argument(
        "--l2_lambda", default=0.1, type=float, help="lambda for l2 norm"
    )
    parser.add_argument(
        "--ewc", default=False, action="store_true", help="whether to use ewc"
    )
    parser.add_argument("--ewc_lambda", default=0.1, type=float, help="lambda for ewc")
    parser.add_argument(
        "--MAS", default=False, action="store_true", help="whether to use mas"
    )
    parser.add_argument("--mas_lambda", default=0.1, type=float, help="lambda for mas")
    parser.add_argument(
        "--si", default=False, action="store_true", help="whether to use si"
    )
    parser.add_argument("--si_c", default=0.1, type=float, help="c for si")
    parser.add_argument(
        "--online", default=False, action="store_true", help="whether to use online"
    )
    parser.add_argument(
        "--replay", default=False, action="store_true", help="whether to use replay"
    )
    parser.add_argument(
        "--n_fisher_sample", default=None, type=int, help="number of fisher sample"
    )
    parser.add_argument(
        "--retrain", default=False, action="store_true", help="whether to retrain"
    )
    parser.add_argument(
        "--LIRF", default=False, action="store_true", help="whether to use LIRF"
    )
    parser.add_argument("--LIRF_T", default=10, type=float, help="lambda for LIRF")
    parser.add_argument("--LIRF_alpha", default=0.1, type=float, help="lambda for LIRF")
    # SCRUB method
    parser.add_argument(
        "--SCRUB", default=False, action="store_true", help="whether to use SCRUB"
    )
    parser.add_argument(
        "--sgda_smoothing", default=0.0, type=float, help="smoothing for sgda"
    )
    parser.add_argument("--sgda_gamma", default=0.99, type=float, help="gamma for sgda")
    parser.add_argument(
        "--sgda_alpha", default=0.001, type=float, help="alpha for sgda"
    )
    parser.add_argument(
        "--sgda_learning_rate", default=1e-4, type=float, help="lr for sgda"
    )
    parser.add_argument(
        "--sgda_momentum", default=0.9, type=float, help="momentum for sgda"
    )
    parser.add_argument(
        "--sgda_weight_decay", default=5e-4, type=float, help="weight_decay for sgda"
    )
    parser.add_argument(
        "--SCRUB_superepoch", default=10, type=int, help="superepoch for sgda"
    )
    parser.add_argument(
        "--kd_T", default=2.0, type=float, help="temperature for kd loss"
    )
    parser.add_argument(
        "--scrub_decay_epoch", default=100, type=int, help="decay epoch for sgda"
    )
    # parser.add_argument('--scrub_decay_rate', default=3, type=int, help='warmup epoch for sgda')
    # Lwf method
    parser.add_argument(
        "--Lwf", default=False, action="store_true", help="whether to use Lwf"
    )
    parser.add_argument("--Lwf_T", default=2, type=float, help="temperature for Lwf")
    parser.add_argument(
        "--Lwf_lambda_kd", default=0.5, type=float, help="lambda kd for Lwf"
    )
    parser.add_argument(
        "--Lwf_lambda_remain", default=1, type=float, help="lambda remain for Lwf"
    )
    # DER method
    parser.add_argument(
        "--Der", default=False, action="store_true", help="whether to use DER"
    )
    parser.add_argument("--DER_lambda", default=0.1, type=float, help="lambda for DER")
    parser.add_argument(
        "--DER_plus", default=False, action="store_true", help="whether to use DER_plus"
    )
    parser.add_argument(
        "--DER_plus_lambda", default=0.1, type=float, help="lambda for DER_plus"
    )
    # FDR method
    parser.add_argument(
        "--FDR", default=False, action="store_true", help="whether to use FDR"
    )
    parser.add_argument("--FDR_lambda", default=0.1, type=float, help="lambda for FDR")
    # CL args
    parser.add_argument("--num_tasks", default=9, type=int, help="number of tasks")
    parser.add_argument("--cl_beta_list", nargs="*", default=[], type=float)
    # FFN freeze args
    parser.add_argument(
        "--ffn_open", default=False, action="store_true", help="whether to freeze ffn"
    )
    parser.add_argument(
        "--only_ffn",
        default=False,
        action="store_true",
        help="whether to train only ffn",
    )
    args = parser.parse_args()
    a=1
    return args