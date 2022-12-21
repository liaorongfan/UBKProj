#! /usr/bin/env python
from src.tools.common import parse_args, default_setup
from src.config.config import get_cfg
from src.engine.trainer import TemplateTrainer


def setup(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # cfg.freeze()
    default_setup(args, cfg)
    return cfg


def main():
    args = parse_args()
    cfg = setup(args)
    runner = TemplateTrainer(cfg)
    if args.test_only:
        return runner.test()
    runner.run()


if __name__ == "__main__":
    main()
