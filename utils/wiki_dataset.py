def load_wiki_dataset(filename, graph=None, verbose_remove=False):
    triplets = []
    cnt_removed = 0
    with open(filename, 'r') as f:
        for i, line in enumerate(f):
            triplets.append([x for x in line.split()])
            assert len(triplets[-1]) == 3, f'Line {i} in {filename} is not a valid triple: {line}'
            if graph is not None:
                try:
                    triplets[-1][0] = graph.entity2id[triplets[-1][0]]
                    triplets[-1][1] = graph.relation2id[triplets[-1][1]]
                    triplets[-1][2] = graph.entity2id[triplets[-1][2]]
                except KeyError:
                    cnt_removed += 1
                    if verbose_remove:
                        print(triplets.pop(), 'was removed')

    if cnt_removed > 0:
        print(f'{cnt_removed} triplets were removed from {filename}')

    return triplets
