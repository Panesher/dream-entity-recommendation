import yaml
import json
import requests
from datasets import load_dataset
from tqdm import tqdm
import pickle
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split


with open('config.yaml') as f:
    config = yaml.safe_load(f)


def download_dataset():
    ds_name_or_path = config['dataset']['name']
    if Path(ds_name_or_path).exists() and 'dream' in str(ds_name_or_path).lower():
        with open(ds_name_or_path, 'r') as f:
            ds = json.load(f)
        train, val = train_test_split(ds, train_size=0.8, random_state=42)
        val, test = train_test_split(val, train_size=0.5, random_state=42)
        return {
            'train': train,
            'validation': val,
            'test': test,
        }
    elif Path(ds_name_or_path).exists() and 'topical' in str(ds_name_or_path).lower():
        ds_name_or_path = Path(ds_name_or_path)
        with open(ds_name_or_path / 'train.json', 'r') as f:
            train = json.load(f)
        with open(ds_name_or_path / 'test.json', 'r') as f:
            test = json.load(f)
        with open(ds_name_or_path / 'valid.json', 'r') as f:
            val = json.load(f)
        return {
            'train': train,
            'validation': val,
            'test': test,
        }

    return load_dataset(config['dataset']['name'])


def extract_sentences_daily_dialog(ds):
    sentences = []
    for data in ds:
        dialog = []
        for message in data['dialog']:
            dialog += [[message.lower().strip()]]
        if dialog:
            sentences += [dialog]
    return sentences


def extract_sentences_topical_chat(ds):
    sentences = []
    for _, v in ds.items():
        dialog = []
        for message_wrapped in v['content']:
            dialog += [message_wrapped['message'].lower().strip()]
        if dialog:
            sentences += [dialog]
    return sentences


def extract_sentences(ds):
    if isinstance(ds, list) and isinstance(ds[0], list) and isinstance(ds[0][0], str):
        return ds
    if isinstance(ds, dict):
        for k in ds:
            if 'content' in ds[k]:
                return extract_sentences_topical_chat(ds)
            break
    return extract_sentences_daily_dialog(ds)


def transform_ed2el(message, ed_res):
    if 'labelled_entities' not in ed_res:
        return {'context': [message]}

    ed_res = ed_res['labelled_entities']
    entity_substr = []
    entity_tags = []
    for entity in ed_res:
        entity_substr += [entity['text']]
        entity_tags += [entity['finegrained_label']]
    return {
        'entity_substr': [entity_substr],
        'entity_tags': [entity_tags],
        'context': [message],
    }


def preprocess_entity_detection(sentences):
    url = config['entity_detection']['url']

    result = []
    for i, dialog in tqdm(
            enumerate(sentences),
            total=len(sentences),
            desc='Entity Detection',
    ):
        data = {'sentences': dialog}
        try:
            ed_res = requests.post(url, json=data).json()
            dialog_res = []
            for d, r in zip(dialog, ed_res):
                dialog_res += [transform_ed2el(d, r)]

            result += [dialog_res]
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f'Error in dialog {i}: {e}')
            result += [[]]

    return result


def preprocess_entity_linking(ed_results):
    url = config['entity_linking']['url']
    result = []
    for i, dialog in tqdm(
            enumerate(ed_results),
            total=len(ed_results),
            desc='Entity Linking',
    ):
        dialog_result = []
        for message in dialog:
            message_result = []
            if 'entity_substr' not in message:
                dialog_result += [[]]
                continue
            try:
                message['context'] = [message['context']]
                el = requests.post(url, json=message).json()
                for entity in el[0]:
                    if entity['entity_ids'] == ['not in wiki']:
                        continue
                    message_result += [
                        {
                            'pages_titles': entity['pages_titles'],
                            'entity_ids': entity['entity_ids'],
                            'confidences': entity['confidences'],
                            'tokens_match_conf': entity['tokens_match_conf'],
                        },
                    ]
                dialog_result += [message_result]
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f'Error in dialog {i}: {e}')
                dialog_result += [[]]

        result += [dialog_result]
    return result


def select_entity_ids(el_message, reference):
    if not el_message:
        return None

    entities = []
    for entity in el_message:
        ids = np.array(entity['entity_ids'])
        confidences = np.array(entity['confidences'])
        filter_ids = list(e_id in reference for e_id in ids)
        if not any(filter_ids):
            continue

        entities.append(ids[filter_ids][np.argmax(confidences[filter_ids])])

    return entities


def preprocess_entity_embeddings(el_results, entity2id, entity_embeddings):
    wikidata_results = {'entity_embeddings': [], 'entity_ids': []}
    for dialog in tqdm(el_results, desc='Entity Embeddings'):
        dialog_entities = []
        dialog_entities_ids = []
        for message in dialog:
            message_entities = []
            message_entities_ids = []
            if not message:
                dialog_entities.append(message_entities)
                dialog_entities_ids.append(message_entities_ids)
                continue

            for entity_id in select_entity_ids(message, entity2id):
                message_entities.append(
                    entity_embeddings[entity2id[entity_id]].tolist(),
                )
                message_entities_ids.append(entity_id)

            dialog_entities.append(message_entities)
            dialog_entities_ids.append(message_entities_ids)

        wikidata_results['entity_embeddings'].append(dialog_entities)
        wikidata_results['entity_ids'].append(dialog_entities_ids)

    return wikidata_results


def get_el_results(el_results, reference):
    wikidata_results = {'entity_ids': []}
    for dialog in tqdm(el_results, desc='Entity Embeddings'):
        dialog_entities_ids = []
        for message in dialog:
            message_entities_ids = []
            if not message:
                dialog_entities_ids.append(message_entities_ids)
                continue

            for entity_id in select_entity_ids(message, reference):
                message_entities_ids.append(entity_id)

            dialog_entities_ids.append(message_entities_ids)

        wikidata_results['entity_ids'].append(dialog_entities_ids)

    return wikidata_results


def get_needed_embedings():
    with open(config['wikidata']['transe'], 'rb') as f:
        transe = pickle.load(f)

    return transe.graph.entity2id, transe.solver.entity_embeddings


if __name__ == '__main__':
    dataset = download_dataset()
    entity2id, _ = get_needed_embedings()

    postprocess_dir = Path(config['dataset']['postprocess']['train']).parent
    for name, value in dataset.items():
        sentences = extract_sentences(value)[:config['dataset']['limit'][name]]
        ed_results = preprocess_entity_detection(sentences)
        with open(postprocess_dir / f"ed_results_{name}.json", "w") as f:
            json.dump(ed_results, f)

        with open(postprocess_dir / f"ed_results_{name}.json", "r") as f:
            ed_results = json.load(f)
        el_results = preprocess_entity_linking(ed_results)
        with open(postprocess_dir / f"el_results_{name}.json", "w") as f:
            json.dump(el_results, f)

        with open(postprocess_dir / f'el_results_{name}.json', 'r') as f:
            el_results = json.load(f)
        el_single_results = get_el_results(el_results, entity2id)
        with open(postprocess_dir / f'el_single_results_{name}.json', 'w') as f:
            json.dump(el_single_results, f)

        # wikidata_encoder_name = Path(config['wikidata']['encoder']).stem.split('_')[0]

        # wikidata_results = preprocess_entity_embeddings(el_results, entity2id, entity_embeddings)
        # with open(f"data/{wikidata_encoder_name}_results_{name}.json", "w") as f:
        #     json.dump(wikidata_results, f)
