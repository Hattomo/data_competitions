# -*- coding: utf-8 -*-


from omegaconf import DictConfig, OmegaConf
from hydra import compose, initialize, initialize_config_dir

if __name__ == "__main__":
    with initialize(version_base=None, config_path="conf", job_name="test_app"):
        cfg1 = compose("default")
        print(OmegaConf.to_yaml(cfg1))

    initialize(version_base=None, config_path="conf", job_name="test_app")
    cfg2 = compose(config_name="debug")
    print(OmegaConf.to_yaml(cfg2))

    # merge cng1 and cfg2
    cfg3 = OmegaConf.merge(cfg1, cfg2)
    print(OmegaConf.to_yaml(cfg3))
