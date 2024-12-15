import hydra


from . import generate


@hydra.main(config_path="./config", config_name="config")
def main(cfg):
    generate.main(cfg)


if __name__ == "__main__":
    main()
