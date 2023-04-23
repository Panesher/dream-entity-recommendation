import yaml
import json
import requests
from datasets import load_dataset
from tqdm import tqdm
import pickle
import numpy as np
from pathlib import Path


with open('config.yaml') as f:
    config = yaml.safe_load(f)


def download_dataset():
    ds = load_dataset(config['dataset']['name'])
    return ds


def extract_sentences(ds):
    sentences = []
    for data in ds:
        dialog = []
        for message in data['dialog']:
            dialog += [[message.lower().strip()]]
        if dialog:
            sentences += [dialog]
    return sentences


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

    for name, value in dataset.items():
        # TODO: delete limit of sentences
        # sentences = extract_sentences(value)[:config['dataset']['limit'][name]]
        # ed_results = preprocess_entity_detection(sentences)
        # with open(f"data/ed_results_{name}.json", "w") as f:
        #     json.dump(ed_results, f)
        
        # el_results = preprocess_entity_linking(ed_results)
        # with open(f"data/el_results_{name}.json", "w") as f:
        #     json.dump(el_results, f)

        with open(f'dataset/el_results_{name}.json', 'r') as f:
            el_results = json.load(f)
        el_single_results = get_el_results(el_results, entity2id)
        with open(f'dataset/el_single_results_{name}.json', 'w') as f:
            json.dump(el_single_results, f)

        # wikidata_encoder_name = Path(config['wikidata']['encoder']).stem.split('_')[0]

        # wikidata_results = preprocess_entity_embeddings(el_results, entity2id, entity_embeddings)
        # with open(f"data/{wikidata_encoder_name}_results_{name}.json", "w") as f:
        #     json.dump(wikidata_results, f)
