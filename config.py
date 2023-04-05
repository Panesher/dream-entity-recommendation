from pathlib import Path
import sys
import yaml

sys.path.append(str(Path('third_party') / 'joint_kg_recommender'))
with open('config.yaml', 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
