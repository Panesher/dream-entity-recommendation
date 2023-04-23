from config import config
import pickle
import json
from model_utils.thswad import THWADModel, load_model_from_transe
from model_utils.trainer import THSWADTrainer
from utils.ds import *
from torch.utils.data import DataLoader

BS = 256
VAL_BS = 32


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


def optimized_transe_load():
    with open(config['wikidata']['transe'], "rb") as f:
        transe = pickle.load(f)
    return load_model_from_transe(transe, id2items, prev_items_total=5), transe.graph


model, graph = optimized_transe_load()
print('model loaded')

ds = DialogDataset(el_single_results['entity_ids'], len(items2id), 5, items2id)
wds = load_wiki_dataset(config['wikidata']['dataset']['train'], graph)

dl = DataLoader(ds, batch_size=BS, shuffle=False)
wdl = DataLoader(wds, batch_size=BS, shuffle=False)
train_loader = TSHWADLoader(wdl, dl)
print('dataset loaded')

opt = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(opt, 1, gamma=0.9)

trainer = THSWADTrainer(
    margin=.5,
    norm_lambda=.2,
    clipping_max_value=5.,
    model_target=.5,
    kg_lambda=.1,
    model=model,
    optimizer=opt,
    scheduler=scheduler,
    train_data_loader=train_loader,
    val_data_loader=train_loader,
    test_data_loader=train_loader,
    verbose=False,
)

print('start training')
trainer.train(20)


print('start eval')
dl_v = DataLoader(ds, batch_size=VAL_BS, shuffle=False)
wdl_v = DataLoader([], batch_size=VAL_BS, shuffle=False)
val_loader = TSHWADLoader(wdl, dl)

print(f'eval accuracy: {trainer.test(val_loader)}')
