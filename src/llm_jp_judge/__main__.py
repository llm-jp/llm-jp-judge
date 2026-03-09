import hydra
from omegaconf import DictConfig

import llm_jp_judge.generate as generate


@hydra.main(config_path="./config", config_name="config")
def main(cfg: DictConfig):
    generate.main(cfg)


if __name__ == "__main__":
    main()
