import yaml
from trainer import Trainer

if __name__ == "__main__":
    with open("experiments/logistic_config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    trainer = Trainer(cfg)
    trainer.train()