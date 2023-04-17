import torch
from torch import nn
from torch.utils.data import DataLoader
from model_utils.thswad import THWADModel, to_gpu
import wandb
from torch.autograd import Variable as V
from tqdm import tqdm
from pathlib import Path
import third_party.joint_kg_recommender.jTransUP.utils.loss as loss


def rec_to_gpu(value):
    if hasattr(value, 'cuda'):
        return to_gpu(value)
    elif isinstance(value, dict):
        return {k: rec_to_gpu(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [rec_to_gpu(v) for v in value]
    else:
        return value


class BaseTrainer(object):
    def __init__(
            self,
            model: THWADModel,
            optimizer: torch.optim.Optimizer,
            scheduler,
            train_data_loader: DataLoader,
            val_data_loader: DataLoader,
            test_data_loader: DataLoader,
            log_interval: int = 5,
            save_interval: int = 10,
            save_path: str = 'model_checkpoints',
            verbose: bool = True,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.test_data_loader = test_data_loader
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.save_path = Path(save_path)
        self.verbose = verbose

    def _train_step(self, batch_idx, batch):
        raise NotImplementedError

    def train_epoch(self, dataloader):
        self.model.train()
        for batch_idx, batch in enumerate(
                tqdm(dataloader, total=len(dataloader), desc='Training'),
        ):
            self.optimizer.zero_grad()
            batch = rec_to_gpu(batch)
            loss_dict = self._train_step(batch_idx, batch)
            self.optimizer.step()
            self.scheduler.step()
            if self.verbose and batch_idx % self.log_interval == 0:
                wandb.log({'train': loss_dict})

    def save_model(self, epoch):
        self.model.save(self.save_path / f'model.pt')

    def train(self, epochs):
        for epoch in range(epochs):
            self.train_epoch(self.train_data_loader)
            if self.save_path and epoch % self.save_interval == self.save_interval - 1:
                self.save_model(epoch)

        self.save_model(epochs)

    def _mean_metrics(self, losses_list_dict):
        mean_loss = {}
        for loss_name, loss_values in losses_list_dict.items():
            mean_loss[loss_name] = torch.mean(torch.stack(loss_values))

        return mean_loss

    def _add_metric(self, loss_dict, single_loss_dict):
        for loss_name, loss_value in single_loss_dict.items():
            if loss_name not in loss_dict:
                loss_dict[loss_name] = []
            loss_dict[loss_name].append(loss_value)

        return loss_dict

    def _test_step(self, batch_idx, batch):
        raise NotImplementedError

    @torch.no_grad()
    def test(self, dataloader):
        self.model.eval()
        mean_loss = {}
        for batch_idx, batch in enumerate(
                tqdm(dataloader, total=len(dataloader), desc='Validating'),
        ):
            batch = rec_to_gpu(batch)
            loss_dict = self._test_step(batch_idx, batch)
            mean_loss = self._add_metric(mean_loss, loss_dict)

        mean_loss = self._mean_metrics(mean_loss)
        if self.verbose:
            wandb.log({'test': mean_loss})
        return mean_loss


class THSWADTrainer(BaseTrainer):
    def __init__(self, margin, norm_lambda, clipping_max_value, model_target, kg_lambda, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kg_lambda = kg_lambda
        self.model_target = model_target
        self.clipping_max_value = clipping_max_value
        self.norm_lambda = norm_lambda
        self.margin = margin

        self.rec_total_loss = 0
        self.kg_total_loss = 0

    def _make_negative_rating(self, next_items):
        return to_gpu(torch.randint(0, len(self.model.item2entity), (len(next_items),)))

    def _make_negative_triple(self, heads, tails):
        mask = torch.rand(len(heads)) > .5
        heads[mask] = to_gpu(torch.randint(0, len(self.model.entity2id), (len(heads[mask]),)))
        tails[~mask] = to_gpu(torch.randint(0, len(self.model.entity2id), (len(tails[~mask]),)))
        return heads, tails

    def _train_step(self, batch_idx, batch):
        if batch['is_rec']:
            prevs, next_items = batch['ratings']
            neg_items = self._make_negative_rating(next_items)

            # Run model. output: batch_size * cand_num, input: ratings, triples, is_rec=True
            pos_score = self.model((prevs, next_items), None, is_rec=True)
            neg_score = self.model((prevs, neg_items), None, is_rec=True)

            # Calculate loss.
            losses = loss.bprLoss(pos_score, neg_score, target=self.model_target)
            losses += loss.orthogonalLoss(self.model.pref_embeddings.weight, self.model.pref_norm_embeddings.weight)
        # kg train
        else:
            h, r, t = batch['triplets']
            nh, nt = self._make_negative_triple(h, t)

            # Run model. output: batch_size * cand_nu, input: ratings, triples, is_rec=True
            pos_score = self.model(None, (h, r, t), is_rec=False)
            neg_score = self.model(None, (nh, r, nt), is_rec=False)

            # Calculate loss.
            losses = loss.marginLoss()(pos_score, neg_score, self.margin)

            ent_embeddings = self.model.ent_embeddings(torch.cat([h, t, nh, nt]))
            rel_embeddings = self.model.rel_embeddings(r) * 2
            norm_embeddings = self.model.norm_embeddings(r) * 2
            losses += loss.orthogonalLoss(rel_embeddings, norm_embeddings)

            losses = losses + loss.normLoss(ent_embeddings) + loss.normLoss(rel_embeddings)
            losses = self.kg_lambda * losses

        # Backward pass.
        losses.backward()

        # Hard Gradient Clipping
        nn.utils.clip_grad_norm_([param for name, param in self.model.named_parameters()], self.clipping_max_value)

        if batch['is_rec']:
            self.rec_total_loss += losses.item()
        else:
            self.kg_total_loss += losses.item()

    def _test_step(self, batch_idx, batch):
        if batch['is_rec']:
            prevs, next_items = batch['ratings']
            score = self.model.evaluate_rec(prevs)
            accuracy = (score.argmax(dim=1) == next_items).float().mean()
            return {'accuracy': accuracy}
        return {}
        # else:
        #     h, r, t = batch['triplets']
        #     head_score = self.model.evaluate_head(t, r)
        #     tail_score = self.model.evaluate_tail(h, r)
        #     head_accuracy = (head_score.argmax(dim=1) == h).float().mean()
        #     tail_accuracy = (tail_score.argmax(dim=1) == t).float().mean()
        #     return {'head_accuracy': head_accuracy, 'tail_accuracy': tail_accuracy}
