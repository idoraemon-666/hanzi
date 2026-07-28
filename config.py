import configargparse
import argparse
import json

def config_parser():
    parser = configargparse.ArgumentParser()
    
    parser.add_argument("--config", is_config_file=True, help="config file path")
    parser.add_argument("--model_name", type=str, default="rnn256_softplus")
    parser.add_argument("--experiment", type=str, default="train_2link_multi")
    parser.add_argument("--protocol_config", type=str)

    return parser


def load_protocol_config(path):
    with open(path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    protocol = config.get("protocol", config.get("project"))
    if protocol not in {
        "digit_writing_original_protocol2",
        "hanzi_stroke_temporal_composition",
    }:
        raise ValueError("unknown project protocol configuration")
    return config


def run_protocol_config(path):
    config = load_protocol_config(path)
    if config.get("project") == "hanzi_stroke_temporal_composition":
        from hanzi_writing.training import run_config

        return run_config(config)
    run_kind = config["run_kind"]
    if run_kind == "base_training":
        from train import train_digit_base_model

        return train_digit_base_model(config)
    if run_kind == "composition":
        from digit_writing.experiments import run_composition_config

        return run_composition_config(config)
    if run_kind == "transfer5":
        from digit_writing.experiments import run_transfer_config

        return run_transfer_config(config)
    raise ValueError(f"unsupported run_kind: {run_kind}")


if __name__ == "__main__":
    arguments = config_parser().parse_args()
    if arguments.protocol_config is None:
        raise ValueError("--protocol_config is required for a protocol run")
    run_protocol_config(arguments.protocol_config)
