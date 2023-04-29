from config import config
import pickle
import json
from model_utils.thswad import THWADModel, load_model_from_transe
from model_utils.trainer import THSWADTrainer
from utils.ds import *
from utils.lrs import get_epoch_lr
from torch.utils.data import DataLoader
import wandb

wandb.init(
    project="dream-entiry-recommendation", 
    name=f'first_run',
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
    },
)

BS = wandb.config['train_batch_size']
VAL_BS = wandb.config['val_batch_size']


with open(config['dataset']['postprocess']['train'], 'r') as f:
    el_single_results = json.load(f)

ds_ = DialogDataset(el_single_results['entity_ids'], 'pad', 5)
items2id = {}
for i, (prev, next_id) in enumerate(ds_):
    for item in prev:
        if item not in items2id and item != 'pad':
            items2id[item] = len(items2id)
    if next_id not in items2id and next_id != 'pad':
        items2id[next_id] = len(items2id)
id2items = list(items2id.keys())
ds = DialogDataset(el_single_results['entity_ids'], len(items2id), 5, items2id)

with open(config['dataset']['postprocess']['val'], 'r') as f:
    el_val = json.load(f)
ds_val = DialogDataset(el_val['entity_ids'], len(items2id), 5, items2id)
dl_v = DataLoader(ds_val, batch_size=VAL_BS, shuffle=False)
wdl_v = DataLoader([], batch_size=VAL_BS, shuffle=False)
val_loader = TSHWADLoader(wdl_v, dl_v)

def optimized_transe_load():
    with open(config['wikidata']['transe'], "rb") as f:
        transe = pickle.load(f)
    return load_model_from_transe(transe, id2items, prev_items_total=5), transe.graph


model, graph = optimized_transe_load()
# model.load('model_checkpoints/model_1.pt')

print('model loaded')

wds = load_wiki_dataset(config['wikidata']['dataset']['train'], graph)
dl = DataLoader(ds, batch_size=BS, shuffle=False)
wdl = DataLoader(wds, batch_size=BS, shuffle=True)
train_loader = TSHWADLoader(wdl, dl)
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
