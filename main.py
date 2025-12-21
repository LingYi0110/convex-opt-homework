from pathlib import Path
import yaml
from trainer import Trainer


experiment_path = r'experiments/logistic/l2/logistic_l2_train_7.yaml'

def train_from_config(config_path):
    print(f'Starting training: {config_path}')
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    trainer = Trainer(cfg)
    trainer.train()

if __name__ == "__main__":
    experiment_path = Path(experiment_path)

    if experiment_path.is_dir():
        yaml_files = sorted(experiment_path.rglob('*.yaml')) + sorted(experiment_path.rglob('*.yml'))
        if not yaml_files:
            raise FileNotFoundError("No YAML configuration files found in the experiments directory.")

        for yaml_file in yaml_files:
            train_from_config(yaml_file)

    elif experiment_path.is_file():
        train_from_config(experiment_path)
    else:
        raise FileNotFoundError("The specified experiment path does not exist.")
