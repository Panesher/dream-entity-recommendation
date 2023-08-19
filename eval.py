from config import config
import pickle
import json
from model_utils.thswad import load_model_from_transe, load_light_model, load_light_model_from_path
from model_utils.trainer import THSWADTrainer, BaseTrainer
from utils.ds import *
from utils.lrs import get_epoch_lr
from torch.utils.data import DataLoader
import wandb

wandb_config={
    'margin': .5,
    'norm_lambda': 1.,
    'clipping_max_value': 5.,
    'model_target': .5,
    'kg_lambda': .1,
    'train_batch_size': 256,
    'val_batch_size': 1,
    'max_epochs': 15,
    'metric_tolerance': .005,
    'eraly_stopping': 10,
    'lr': .005,
    'prev_count': 5,
    'use_st_gumbel': False,
    'l1_flag': True,
    'use_item_emb': True,
    'dataset': config['dataset']['postprocess']['train'],
}

BS = wandb_config['train_batch_size']
VAL_BS = wandb_config['val_batch_size']
PREV_CNT = wandb_config['prev_count']


def make_item2id(el_single_train):
    ds_ = DialogDataset(el_single_train['entity_ids'], 'pad', PREV_CNT)
    items2id = {}
    for i, (prev, next_id) in enumerate(ds_):
        for item in prev:
            if item not in items2id and item != 'pad':
                items2id[item] = len(items2id)
        if next_id not in items2id and next_id != 'pad':
            items2id[next_id] = len(items2id)
    return items2id

with open(config['dataset']['postprocess']['train'], 'r') as f:
    el_single_results = json.load(f)
items2id = make_item2id(el_single_results)
id2items = list(items2id.keys())


def make_dataloader(el_single_res, wikidata_ds, items2id=items2id, batch_size=BS, shuffle=True):
    if el_single_res:
        ds = DialogDataset(el_single_res['entity_ids'], len(items2id), PREV_CNT, items2id)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
    else:
        dl = DataLoader([], batch_size=batch_size, shuffle=False)
    if wikidata_ds:
        wdl = DataLoader(wikidata_ds, batch_size=batch_size, shuffle=shuffle)
    else:
        wdl = DataLoader([], batch_size=batch_size, shuffle=False)
    return TSHWADLoader(wdl, dl)


with open(config['dataset']['postprocess']['val'], 'r') as f:
    el_val = json.load(f)
val_loader = make_dataloader(el_val, [], batch_size=VAL_BS, shuffle=False)

model = load_light_model_from_path(config['model']['light_path'], PREV_CNT)

print('model loaded')

opt = torch.optim.Adam(model.parameters(), lr=wandb_config['lr'])
scheduler = get_epoch_lr(opt, 10, wandb_config['max_epochs'] * 10, start_value=wandb_config['lr'])

trainer = THSWADTrainer(
    margin=wandb_config['margin'],
    norm_lambda=wandb_config['norm_lambda'],
    clipping_max_value=wandb_config['clipping_max_value'],
    model_target=wandb_config['model_target'],
    kg_lambda=wandb_config['kg_lambda'],
    model=model,
    optimizer=opt,
    scheduler=scheduler,
    metric_tolerance=wandb_config['metric_tolerance'],
    eraly_stopping=wandb_config['eraly_stopping'],
    verbose=False,
)
print(trainer.test(val_loader))
