import json
import yaml 
from typing import List, Dict


def merge_datasets(datasets_paths: List[Dict[str, str]], merged_dataset_path: Dict[str, str]):
    merged_train = []
    merged_val = []
    merged_test = []
    for paths in datasets_paths:
        with open(paths['train'], 'r') as f:
            merged_train.extend(json.load(f)['entity_ids'])
        with open(paths['val'], 'r') as f:
            merged_val.extend(json.load(f)['entity_ids'])
        with open(paths['test'], 'r') as f:
            merged_test.extend(json.load(f)['entity_ids'])

    with open(merged_dataset_path['train'], 'w') as f:
        json.dump({'entity_ids': merged_train}, f)
    with open(merged_dataset_path['val'], 'w') as f:
        json.dump({'entity_ids': merged_val}, f)
    with open(merged_dataset_path['test'], 'w') as f:
        json.dump({'entity_ids': merged_test}, f)


if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    merge_datasets(config['dataset']['merge_paths'], config['dataset']['postprocess'])
