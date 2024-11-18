import hydra


from . import inference


@hydra.main(config_path="./config", config_name="config")
def main(cfg):
    inference.main(cfg)


if __name__ == "__main__":
    main()
