import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as V
from pathlib import Path

from third_party.joint_kg_recommender.jTransUP.utils.misc import (
    to_gpu,
    projection_transH_pytorch,
)


def make_embedding(num_embeddings, embedding_dim, padding=False):
    tensor = torch.FloatTensor(
        num_embeddings - 1 if padding else num_embeddings, embedding_dim,
    )

    nn.init.xavier_uniform_(tensor)
    norm_ent_weight = F.normalize(tensor, p=2, dim=1)
    weight = nn.Parameter(
        torch.cat([norm_ent_weight, torch.zeros(1, embedding_dim)], dim=0)
        if padding
        else norm_ent_weight,
    )

    embedding = nn.Embedding(
        num_embeddings,
        embedding_dim,
        padding_idx=num_embeddings - 1 if padding else None,
    )
    embedding.weight = weight
    embedding.weight.data = F.normalize(embedding.weight.data, p=2, dim=1)
    return to_gpu(embedding)


def reverse_dict(dictionary):
    if isinstance(dictionary, dict):
        return {v: k for k, v in dictionary.items()}
    elif isinstance(dictionary, list):
        return {v: k for k, v in enumerate(dictionary)}
    else:
        raise NotImplementedError(
            f'dictionary should be dict or list: {type(dictionary)}',
        )


# TransH with added dependencies model
class THWADModel(nn.Module):
    def __init__(
            self,
            item2entity,
            id2entity,
            embedding_size,
            item_total,
            entity_total,
            relation_total,
            prev_items_total,
            rel2id=None,
            l1_flag=False,
            is_share=False,
            use_st_gumbel=False,
    ):
        super().__init__()
        self.l1_flag = l1_flag
        self.is_share = is_share
        self.use_st_gumbel = use_st_gumbel

        self.embedding_size = embedding_size
        self.item_total = item_total + 1
        # padding when item are not aligned with any entity
        self.ent_total = entity_total + 1
        self.rel_total = relation_total

        # store item to item-entity dic
        self.item2entity = item2entity
        self.entity2item = reverse_dict(item2entity)

        self.id2entity = id2entity
        self.entity2id = reverse_dict(id2entity)

        self.rel2id = rel2id if rel2id else None
        self.id2rel = reverse_dict(rel2id) if rel2id else None

        # transup
        # init item embeddings
        self.item_embeddings = make_embedding(
            self.item_total, self.embedding_size, padding=True,
        )
        # init preference parameters
        self.pref_embeddings = make_embedding(
            self.rel_total, self.embedding_size,
        )
        self.pref_norm_embeddings = make_embedding(
            self.rel_total, self.embedding_size,
        )

        # init transh
        self.ent_embeddings = make_embedding(
            self.ent_total, self.embedding_size, padding=True,
        )
        self.rel_embeddings = make_embedding(
            self.rel_total, self.embedding_size,
        )
        self.norm_embeddings = make_embedding(
            self.rel_total, self.embedding_size,
        )

        # meaner
        self.prev_meaner = to_gpu(
            nn.Parameter(torch.ones(prev_items_total) / prev_items_total),
        )

    def save(self, model_path):
        save_dir = Path(model_path)
        save_dir.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model': self.state_dict(),
            'item2entity': self.item2entity,
            'id2entity': self.id2entity,
            'entity2id': self.entity2id,
            'rel2id': self.rel2id,
            'id2rel': self.id2rel,
        }, model_path)

    def load(self, model_path):
        model_path = Path(model_path)

        # load to cpu because gpu probably contains less memory
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        self = self.cpu()
        self.load_state_dict(checkpoint['model'])
        to_gpu(self)

        self.item2entity = checkpoint['item2entity']
        self.id2entity = checkpoint['id2entity']
        self.entity2id = checkpoint['entity2id']
        self.rel2id = checkpoint['rel2id']
        self.id2rel = checkpoint['id2rel']

    def get_items_name(self, i_ids):
        return [self.id2entity[i_id] for i_id in i_ids]

    def get_entity(self, i_id):
        return self.item2entity.get(i_id, self.ent_total - 1)

    def get_item(self, e_id):
        return self.entity2item.get(e_id, self.item_total - 1)

    def padding_items(self, i_ids):
        padded_e_ids = []
        for i_id in i_ids:
            padded_e_ids.append(self.get_entity(i_id))
        return padded_e_ids

    def get_multi_item_emb(self, ids):
        return self.prev_meaner @ self.item_embeddings(ids)

    def get_item_ent_emb(self, ids=None):
        if ids is None:
            ids = to_gpu(torch.arange(self.item_total))

        e_ids = self.padding_items(ids.data)
        e_var = to_gpu(V(torch.LongTensor(e_ids)))

        return self.item_embeddings(ids) + self.ent_embeddings(e_var)

    def forward(self, ratings, triples, is_rec=True):
        if is_rec and ratings is not None:
            i_prevs, i_next = ratings

            i_prevs_e = self.get_multi_item_emb(i_prevs)
            i_next_e = self.get_item_ent_emb(i_next)

            _, r_e, norm = self.get_preferences(
                i_prevs_e, i_next_e, use_st_gumbel=self.use_st_gumbel,
            )

            proj_i_prev_e = projection_transH_pytorch(i_prevs_e, norm)
            proj_i_next_e = projection_transH_pytorch(i_next_e, norm)

            if self.l1_flag:
                score = torch.sum(
                    torch.abs(proj_i_prev_e + r_e - proj_i_next_e), 1,
                )
            else:
                score = torch.sum(
                    (proj_i_prev_e + r_e - proj_i_next_e) ** 2, 1,
                )
        elif not is_rec and triples is not None:
            h, r, t = triples
            h_e = self.ent_embeddings(h)
            t_e = self.ent_embeddings(t)
            r_e = self.rel_embeddings(r)
            norm_e = self.norm_embeddings(r)

            proj_h_e = projection_transH_pytorch(h_e, norm_e)
            proj_t_e = projection_transH_pytorch(t_e, norm_e)

            if self.l1_flag:
                score = torch.sum(torch.abs(proj_h_e + r_e - proj_t_e), 1)
            else:
                score = torch.sum((proj_h_e + r_e - proj_t_e) ** 2, 1)
        else:
            raise NotImplementedError

        return score

    def evaluate_rec(self, i_prev, all_i_ids=None):
        batch_size = len(i_prev)
        i_next_e = self.get_item_ent_emb(all_i_ids)
        item_total, dim = i_next_e.size()

        i_prev_e = self.get_multi_item_emb(i_prev)
        # expand u and i to pair wise match, batch * item * dim
        i_prev_e = i_prev_e.expand(item_total, batch_size, dim).permute(
            1, 0, 2,
        )
        i_next_e = i_next_e.expand(batch_size, item_total, dim)

        # batch * item * dim
        _, r_e, norm = self.get_preferences(
            i_prev_e, i_next_e, use_st_gumbel=self.use_st_gumbel,
        )

        proj_i_prev_e = projection_transH_pytorch(i_next_e, norm)
        proj_i_next_e = projection_transH_pytorch(i_next_e, norm)

        # batch * item
        if self.l1_flag:
            score = torch.sum(
                torch.abs(proj_i_prev_e + r_e - proj_i_next_e), 2,
            )
        else:
            score = torch.sum((proj_i_prev_e + r_e - proj_i_next_e) ** 2, 2)
        return score

    def evaluate_head(self, t, r, all_e_ids=None):
        batch_size = len(t)
        all_e = (
            self.ent_embeddings(all_e_ids)
            if all_e_ids is not None and self.is_share
            else self.ent_embeddings.weight
        )
        ent_total, dim = all_e.size()
        # batch * dim
        t_e = self.ent_embeddings(t)
        r_e = self.rel_embeddings(r)
        norm_e = self.norm_embeddings(r)

        proj_t_e = projection_transH_pytorch(t_e, norm_e)
        c_h_e = proj_t_e - r_e

        # batch * entity * dim
        c_h_expand = c_h_e.expand(ent_total, batch_size, dim).permute(1, 0, 2)

        # batch * entity * dim
        norm_expand = norm_e.expand(ent_total, batch_size, dim).permute(
            1, 0, 2,
        )

        ent_expand = all_e.expand(batch_size, ent_total, dim)
        proj_ent_e = projection_transH_pytorch(ent_expand, norm_expand)

        # batch * entity
        if self.l1_flag:
            score = torch.sum(torch.abs(c_h_expand - proj_ent_e), 2)
        else:
            score = torch.sum((c_h_expand - proj_ent_e) ** 2, 2)
        return score

    def evaluate_tail(self, h, r, all_e_ids=None):
        batch_size = len(h)
        all_e = (
            self.ent_embeddings(all_e_ids)
            if all_e_ids is not None and self.is_share
            else self.ent_embeddings.weight
        )
        ent_total, dim = all_e.size()
        # batch * dim
        h_e = self.ent_embeddings(h)
        r_e = self.rel_embeddings(r)
        norm_e = self.norm_embeddings(r)

        proj_h_e = projection_transH_pytorch(h_e, norm_e)
        c_t_e = proj_h_e + r_e

        # batch * entity * dim
        c_t_expand = c_t_e.expand(ent_total, batch_size, dim).permute(1, 0, 2)

        # batch * entity * dim
        norm_expand = norm_e.expand(ent_total, batch_size, dim).permute(
            1, 0, 2,
        )

        ent_expand = all_e.expand(batch_size, ent_total, dim)
        proj_ent_e = projection_transH_pytorch(ent_expand, norm_expand)

        # batch * entity
        if self.l1_flag:
            score = torch.sum(torch.abs(c_t_expand - proj_ent_e), 2)
        else:
            score = torch.sum((c_t_expand - proj_ent_e) ** 2, 2)
        return score

    # u_e, i_e : batch * dim or batch * item * dim
    def get_preferences(self, u_e, i_e, use_st_gumbel=False):
        # use item and user embedding to compute preference distribution
        # pre_probs: batch * rel, or batch * item * rel
        pre_probs = (
            torch.matmul(
                u_e + i_e,
                torch.t(
                    self.pref_embeddings.weight + self.rel_embeddings.weight,
                ),
            )
            / 2
        )
        if use_st_gumbel:
            pre_probs = self.st_gumbel_softmax(pre_probs)

        r_e = (
            torch.matmul(
                pre_probs,
                self.pref_embeddings.weight + self.rel_embeddings.weight,
            )
            / 2
        )
        norm = (
            torch.matmul(
                pre_probs,
                self.pref_norm_embeddings.weight + self.norm_embeddings.weight,
            )
            / 2
        )

        return pre_probs, r_e, norm

    # batch or batch * item
    def convert_to_one_hot(self, indices, num_classes):
        """
        Args:
            indices (Variable): A vector containing indices,
                whose size is (batch_size,).
            num_classes (Variable): The number of classes, which would be
                the second dimension of the resulting one-hot matrix.
        Returns:
            result: The one-hot matrix of size (batch_size, num_classes).
        """

        old_shape = indices.shape
        new_shape = torch.Size([i for i in old_shape] + [num_classes])
        indices = indices.unsqueeze(len(old_shape))

        one_hot = V(
            indices.data.new(new_shape)
            .zero_()
            .scatter_(len(old_shape), indices.data, 1),
        )
        return one_hot

    def masked_softmax(self, logits):
        eps = 1e-20
        probs = F.softmax(logits, dim=len(logits.shape) - 1)
        return probs

    def st_gumbel_softmax(self, logits, temperature=1.0):
        """
        Return the result of Straight-Through Gumbel-Softmax Estimation.
        It approximates the discrete sampling via Gumbel-Softmax trick
        and applies the biased ST estimator.
        In the forward propagation, it emits the discrete one-hot result,
        and in the backward propagation it approximates the categorical
        distribution via smooth Gumbel-Softmax distribution.
        Args:
            logits (Variable): A un-normalized probability values,
                which has the size (batch_size, num_classes)
            temperature (float): A temperature parameter. The higher
                the value is, the smoother the distribution is.
        Returns:
            y: The sampled output, which has the property explained above.
        """

        eps = 1e-20
        u = logits.data.new(*logits.size()).uniform_()
        gumbel_noise = V(-(torch.log(-(torch.log(u + eps)) + eps)))
        y = logits + gumbel_noise
        y = self.masked_softmax(logits=y / temperature)
        y_argmax = y.max(len(y.shape) - 1)[1]
        y_hard = self.convert_to_one_hot(
            indices=y_argmax, num_classes=y.size(len(y.shape) - 1),
        ).float()
        y = (y_hard - y).detach() + y
        return y

    def report_preference(self, i_prevs, i_next_ids):
        item_num = len(i_next_ids)
        # item * dim
        i_prevs_e = self.get_multi_item_emb(i_prevs.expand(item_num))
        i_next_e = self.get_item_ent_emb(i_next_ids)

        return self.get_preferences(
            i_prevs_e, i_next_e, use_st_gumbel=self.use_st_gumbel,
        )

    def disable_grad(self):
        for name, param in self.named_parameters():
            param.requires_grad = False

    def enable_grad(self):
        for name, param in self.named_parameters():
            param.requires_grad = True


def load_model_from_transe(
        transe, item_ids, add_rel_cnt=1, **model_init_kwargs,
) -> THWADModel:
    _, embedding_size = transe.solver['entity_embeddings'].shape
    relation_total, embedding_size_r = transe.solver[
        'relation_embeddings'
    ].shape
    relation_total += add_rel_cnt
    assert embedding_size == embedding_size_r
    item_total = len(item_ids)
    item2entity = {
        item_id: transe.graph.entity2id[item]
        for item_id, item in enumerate(item_ids)
    }
    model = THWADModel(
        item2entity=item2entity,
        id2entity=transe.graph.id2entity,
        rel2id=transe.graph.id2relation,
        embedding_size=embedding_size,
        item_total=item_total,
        entity_total=1,  # for fast preload
        relation_total=relation_total,
        **model_init_kwargs,
    )
    model.ent_total = transe.solver.entity_embeddings.shape[0] + 1
    model.ent_embeddings = to_gpu(
        nn.Embedding.from_pretrained(
            torch.cat(
                [
                    torch.tensor(transe.solver.entity_embeddings),
                    torch.zeros(1, model.embedding_size),
                ],
                dim=0,
            ),
            padding_idx=model.ent_total - 1,
        ),
    )
    model.rel_embeddings = to_gpu(
        nn.Embedding.from_pretrained(
            torch.cat(
                [
                    torch.tensor(transe.solver.relation_embeddings),
                    torch.zeros(
                        add_rel_cnt, model.embedding_size,
                    ),
                ],
                dim=0,
            ),
        ),
    )
    return model
