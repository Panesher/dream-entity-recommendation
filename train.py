from config import config
import pickle
import json
from model_utils.thswad import load_model_from_transe
from model_utils.trainer import THSWADTrainer
from utils.ds import *
from utils.lrs import get_epoch_lr
from torch.utils.data import DataLoader
import wandb

wandb.init(
    project="dream-entiry-recommendation", 
    name=f'Run with val loss',
    config={
        'margin': .5,
        'norm_lambda': .2,
        'clipping_max_value': 5.,
        'model_target': .5,
        'kg_lambda': .1,
        'train_batch_size': 256,
        'val_batch_size': 4,
        'max_epochs': 30,
        'metric_tolerance': .005,
        'eraly_stopping': 10,
        'lr': .01,
        'prev_count': 5,
        'use_st_gumbel': False,
        'l1_flag': False,
    },
)

BS = wandb.config['train_batch_size']
VAL_BS = wandb.config['val_batch_size']
PREV_CNT = wandb.config['prev_count']


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
    ds = DialogDataset(el_single_res['entity_ids'], len(items2id), PREV_CNT, items2id)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
    wdl = DataLoader(wikidata_ds, batch_size=batch_size, shuffle=shuffle)
    return TSHWADLoader(wdl, dl)


with open(config['dataset']['postprocess']['val'], 'r') as f:
    el_val = json.load(f)
val_loader = make_dataloader(el_val, [], batch_size=VAL_BS, shuffle=False)


def optimized_transe_load():
    with open(config['wikidata']['transe'], "rb") as f:
        transe = pickle.load(f)
    return load_model_from_transe(
        transe,
        id2items,
        prev_items_total=PREV_CNT,
        use_st_gumbel=wandb.config['use_st_gumbel'],
        l1_flag=wandb.config['l1_flag'],
    ), transe.graph


model, graph = optimized_transe_load()

print('model loaded')

train_loader = make_dataloader(el_single_results, load_wiki_dataset(config['wikidata']['dataset']['train'], graph))
print('dataset loaded')

opt = torch.optim.Adam(model.parameters(), lr=wandb.config['lr'])
scheduler = get_epoch_lr(opt, len(train_loader), wandb.config['max_epochs'], start_value=wandb.config['lr'])

trainer = THSWADTrainer(
    margin=wandb.config['margin'],
    norm_lambda=wandb.config['norm_lambda'],
    clipping_max_value=wandb.config['clipping_max_value'],
    model_target=wandb.config['model_target'],
    kg_lambda=wandb.config['kg_lambda'],
    model=model,
    optimizer=opt,
    scheduler=scheduler,
    metric_tolerance=wandb.config['metric_tolerance'],
    eraly_stopping=wandb.config['eraly_stopping'],
    verbose=True,
)

print('start training')
trainer.train(train_loader, wandb.config['max_epochs'], val_dataloader=val_loader)
