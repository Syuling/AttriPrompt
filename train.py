import argparse
import torch

from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer

# custom
import datasets.oxford_pets
import datasets.oxford_flowers
import datasets.fgvc_aircraft
import datasets.dtd
import datasets.eurosat
import datasets.stanford_cars
import datasets.food101
import datasets.sun397
import datasets.caltech101
import datasets.ucf101
import datasets.imagenet

import datasets.imagenet_sketch
import datasets.imagenetv2
import datasets.imagenet_a
import datasets.imagenet_r



import trainers.attriprompt
import trainers.cocoop
import trainers.coop
import trainers.maple
import trainers.baseclip_graph_v1
import trainers.prograd
import trainers.taskres
import trainers.lamm
# import trainers.attriprompt_blip



def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head

    if args.top_k:
        cfg.MODEL.TOP_K = args.top_k

    if args.num_prompt:
        cfg.MODEL.NUM_PROMPT = args.num_prompt
    
    if args.Head:
        cfg.Head = args.Head
    
    if args.cont_dis is not None:
        cfg.cont_dis = args.cont_dis

    if args.alpha is not None:
        cfg.alpha = args.alpha




def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN

    cfg.TRAINER.COOP = CN()
    cfg.TRAINER.COOP.N_CTX = 8  # number of context vectors
    cfg.TRAINER.COOP.CSC = True  # class-specific context
    cfg.TRAINER.COOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COOP.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.COOP.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'
    cfg.TRAINER.COOP.TRAIN_TYPE="freeze"

    cfg.num_prompt = 10
    cfg.top_k = 5
    cfg.PROMPT_DEPTH_VISION = 9
    cfg.PROMPT_DEPTH_TEXT = 9
    cfg.N_CTX_VISION = 4
    cfg.N_CTX_TEXT = 4

    cfg.distributed = False
    cfg.local_rank = 0
    # cfg.rank = 0
    cfg.world_size = 2
    
    cfg.TRAINER.COCOOP = CN()
    cfg.TRAINER.COCOOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COCOOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COCOOP.PREC = "fp16"  # fp16, fp32, amp

    cfg.TRAINER.COOPAL = CN() 
    cfg.TRAINER.COOPAL.METHOD = ""
    cfg.TRAINER.COOPAL.ASPATH = ""
    cfg.TRAINER.COOPAL.AEPATH = ""
    cfg.TRAINER.COOPAL.GAMMA = 0.1
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new

    cfg.TRAINER.MAPLE = CN()
    cfg.TRAINER.MAPLE.N_CTX = 2  # number of context vectors
    cfg.TRAINER.MAPLE.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.MAPLE.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.MAPLE.PROMPT_DEPTH = 9 # Max 12, minimum 0, for 1 it will act as shallow MaPLe (J=1)
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new
    
    cfg.TRAINER.ATTRIPROMPT = CN()
    cfg.TRAINER.ATTRIPROMPT.N_CTX = 8  # number of context vectors
    cfg.TRAINER.ATTRIPROMPT.CSC = True  # class-specific context
    cfg.TRAINER.ATTRIPROMPT.CTX_INIT = ""  # initialization words
    cfg.TRAINER.ATTRIPROMPT.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.ATTRIPROMPT.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'
    cfg.TRAINER.ATTRIPROMPT.TRAIN_TYPE="freeze"
    cfg.TRAINER.ATTRIPROMPT.PROMPT_DEPTH_VISION = 9
    cfg.TRAINER.ATTRIPROMPT.PROMPT_DEPTH_TEXT = 9
    cfg.TRAINER.ATTRIPROMPT.N_CTX_VISION = 4
    cfg.TRAINER.ATTRIPROMPT.N_CTX_TEXT = 4

    cfg.TRAINER.ATPROMPT = CN()
    cfg.TRAINER.ATPROMPT.USE_ATPROMPT = False
    cfg.TRAINER.ATPROMPT.N_ATT1 = 4
    cfg.TRAINER.ATPROMPT.N_ATT2 = 4
    cfg.TRAINER.ATPROMPT.N_ATT3 = 4
    cfg.TRAINER.ATPROMPT.ATT_NUM = 0
    cfg.TRAINER.ATPROMPT.ATT1_TEXT = "none"
    cfg.TRAINER.ATPROMPT.ATT2_TEXT = "none"
    cfg.TRAINER.ATPROMPT.ATT3_TEXT = "none"
    
    cfg.TRAINER.TaskRes = CN()
    cfg.TRAINER.TaskRes.N_CTX = 16  # number of context vectors
    cfg.TRAINER.TaskRes.CSC = False  # class-specific context
    cfg.TRAINER.TaskRes.CTX_INIT = ""  # initialization words
    cfg.TRAINER.TaskRes.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.TaskRes.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'
    cfg.TRAINER.TaskRes.RESIDUAL_SCALE = 1.0
    cfg.TRAINER.TaskRes.ENHANCED_BASE = args.enhanced_base

def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    cfg.freeze()

    return cfg


def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)
    print("Collecting env info ...")
    print("** System info **\n{}\n".format(collect_env_info()))

    trainer = build_trainer(cfg)

    # if args.eval_only:
    #     trainer.load_model(args.model_dir, epoch=args.load_epoch)
    #     trainer.test()
    #     return

    # if not args.no_train:
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        "--seed", type=int, default=-1, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--Head", type=int, default=16, help="only positive value enables a fixed head"
    )
    parser.add_argument(
        "--cont_dis", type=int, default=0, help="continueous distribution or not", 
    )
    parser.add_argument(
        "--alpha", type=float, default=0.0, help="the variance of the continuous distribution"
    )
    parser.add_argument(
        "--num_prompt", type=int, default=10, help="number of prompts"
    )
    parser.add_argument(
        "--top_k", type=int, default=3, help="number of top_k"
    )
    parser.add_argument(
        "--source-domains", type=str, nargs="+", help="source domains for DA/DG"
    )
    parser.add_argument(
        "--target-domains", type=str, nargs="+", help="target domains for DA/DG"
    )
    parser.add_argument(
        "--transforms", type=str, nargs="+", help="data augmentation methods"
    )
    parser.add_argument(
        "--config-file", type=str, default="", help="path to config file"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="",
        help="path to config file for dataset setup",
    )
    parser.add_argument("--trainer", type=str, default="", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "--load-epoch", type=int, help="load model weights at this epoch for evaluation"
    )
    parser.add_argument(
        "--no-train", action="store_true", help="do not call trainer.train()"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    args = parser.parse_args()
    main(args)
