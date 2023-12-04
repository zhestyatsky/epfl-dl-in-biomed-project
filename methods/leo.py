# This code is modified from https://github.com/dragen1860/MAML-Pytorch and https://github.com/katerakelly/pytorch-maml
import numpy as np
import torch
import torch.nn as nn
import wandb

from backbones.blocks import Linear_fw
from methods.meta_template import MetaTemplate


class EncodingNetwork(nn.Module):
    def __init__(self, n_support, n_way, embedding_dim):
        super().__init__()
        self.n_support = n_support
        self.n_way = n_way
        self.embedding_dim = embedding_dim

        self.encoding_layer = nn.Linear(embedding_dim, embedding_dim)
        self.relation_net = nn.Linear(2 * embedding_dim, 2 * embedding_dim)

    def forward(self, x_support):
        encoded_x_support = self.encoding_layer(x_support)

        lhs_relation_net_input = encoded_x_support.unsqueeze(1).tile((1, self.n_way * self.n_support, 1))
        rhs_relation_net_input = encoded_x_support.unsqueeze(0).tile((self.n_way * self.n_support, 1, 1))
        relation_net_input = torch.cat((lhs_relation_net_input, rhs_relation_net_input), dim=-1)

        relation_net_output = self.relation_net(relation_net_input).mean(dim=1)
        relation_net_per_class_output = relation_net_output.view(self.n_way, self.n_support, -1).mean(dim=1)

        means, stds = relation_net_per_class_output.chunk(chunks=2, dim=-1)

        gaussian_vectors = torch.normal(
            torch.zeros(self.n_way, self.embedding_dim),
            torch.ones(self.n_way, self.embedding_dim),
        )

        output = gaussian_vectors * stds + means

        return output


class DecodingNetwork(nn.Module):
    def __init__(self, n_way, embedding_dim, output_dim):
        super().__init__()
        self.n_way = n_way
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim

        self.decoding_layer = nn.Linear(embedding_dim, 2 * output_dim)

    def forward(self, latent_output):
        decoded_output = self.decoding_layer(latent_output)

        means, stds = decoded_output(chunks=2, dim=-1)

        gaussian_vectors = torch.normal(
            torch.zeros(self.n_way, self.output_dim),
            torch.ones(self.n_way, self.output_dim),
        )

        output = gaussian_vectors * stds + means

        return output


class LEO(MetaTemplate):
    def __init__(self, backbone, n_way, n_support, n_task, task_update_num, inner_lr,
                inner_update_step, l2_penalty_coef, kl_coef, orthogonality_penalty_coef, 
                encoder_penalty_coef, approx=False):
            """
            Initialize the LEO (Latent Embedding Optimization) model.

            Args:
                backbone (object): The backbone of the model.
                n_way (int): Number of classes in each task.
                n_support (int): Number of support examples per class.
                n_task (int): Number of tasks.
                task_update_num (int): Number of task update steps.
                inner_lr (float): Inner learning rate for task updates.
                inner_update_step (int): Number of inner update steps.
                l2_penalty_coef (float): Coefficient for L2 penalty.
                kl_coef (float): Coefficient for KL divergence penalty.
                orthogonality_penalty_coef (float): Coefficient for orthogonality penalty.
                encoder_penalty_coef (float): Coefficient for encoder penalty.
                approx (bool, optional): Whether to use first order approximation. Defaults to False.
            """
            super(LEO, self).__init__(backbone, n_way, n_support, change_way=False)

            self.classifier = Linear_fw(self.feat_dim, n_way)
            self.classifier.bias.data.fill_(0)

            if n_way == 1:
                self.type = "regression"
                self.loss_fn = nn.MSELoss()
            else:
                self.type = "classification"
                self.loss_fn = nn.CrossEntropyLoss()

            self.n_task = n_task
            self.task_update_num = task_update_num
            self.inner_lr = inner_lr 
            self.inner_update_step = inner_update_step
            self.l2_penalty_coef = l2_penalty_coef  
            self.kl_coef = kl_coef
            self.orthogonality_penalty_coef = orthogonality_penalty_coef
            self.encoder_penalty_coef = encoder_penalty_coef
            self.approx = approx # first order approximation

            self.encoder = EncodingNetwork(n_support=n_support, n_way=n_way, embedding_dim=self.feat_dim)
            self.decoder = DecodingNetwork(n_way=n_way, embedding_dim=self.feat_dim, output_dim=self.feat_dim+1)

    def forward(self, x):
        out = self.feature.forward(x)
        scores = self.classifier.forward(out)

        # For regression tasks, these are not scores but predictions
        if scores.shape[1] == 1:
            scores = scores.squeeze(1)

        return scores

    def set_forward(self, x, y=None):
        if torch.cuda.is_available():
            x = x.cuda()

        x_support = x[:, :self.n_support, :].contiguous().view(self.n_way * self.n_support, -1)
        x_query = x[:, self.n_support:, :].contiguous().view(self.n_way * self.n_query, -1)

        if y is None:  # Classification task, assign labels (class indices) based on n_way
            y_support = torch.from_numpy(np.repeat(range(self.n_way), self.n_support))
        else:  # Regression task, keep labels as they are
            y_support = y[:, :self.n_support].contiguous().view(self.n_way * self.n_support, -1)

        if torch.cuda.is_available():
            y_support = y_support.cuda()

        self.zero_grad()

        latents_z = self.encoder(x_support)
        weights = self.decoder(latents_z)
        clf_weight, clf_bias = weights.split([self.feat_dim, 1], dim=-1)
        self.classifier.weight.fast = clf_weight
        self.classifier.bias.fast = clf_bias

        for i in range(self.inner_update_step):
            scores = self.forward(x_support)
            set_loss = self.loss_fn(scores, y_support)
            grad = torch.autograd.grad(set_loss, latents_z, create_graph=True)[0]
            latents_z = latents_z - self.inner_lr * grad

            weights = self.decoder(latents_z)
            clf_weight, clf_bias = weights.split([self.feat_dim, 1], dim=-1)
            self.classifier.weight.fast = clf_weight
            self.classifier.bias.fast = clf_bias

        scores = self.forward(x_query)
        return scores

    def set_forward_adaptation(self, x, is_feature=False):  # overwrite parrent function
        raise ValueError('MAML performs further adapation simply by increasing task_upate_num')

    def set_forward_loss(self, x, y=None):
        scores = self.set_forward(x, y)

        if y is None:  # Classification task
            y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
        else:  # Regression task
            y_query = y[:, self.n_support:].contiguous().view(self.n_way * self.n_query, -1)

        if torch.cuda.is_available():
            y_query = y_query.cuda()

        loss = self.loss_fn(scores, y_query)

        return loss

    def train_loop(self, epoch, train_loader, optimizer):  # overwrite parrent function
        print_freq = 10
        avg_loss = 0
        task_count = 0
        loss_all = []
        optimizer.zero_grad()

        # train
        for i, (x, y) in enumerate(train_loader):
            if isinstance(x, list):
                self.n_query = x[0].size(1) - self.n_support
                assert self.n_way == x[0].size(
                    0), f"MAML do not support way change, n_way is {self.n_way} but x.size(0) is {x.size(0)}"
            else:
                self.n_query = x.size(1) - self.n_support
                assert self.n_way == x.size(
                    0), f"MAML do not support way change, n_way is {self.n_way} but x.size(0) is {x.size(0)}"

            # Labels are assigned later if classification task
            if self.type == "classification":
                y = None

            loss = self.set_forward_loss(x, y)
            avg_loss = avg_loss + loss.item()
            loss_all.append(loss)

            task_count += 1

            if task_count == self.n_task:  # MAML update several tasks at one time
                loss_q = torch.stack(loss_all).sum(0)
                loss_q.backward()

                optimizer.step()
                task_count = 0
                loss_all = []
            optimizer.zero_grad()
            if i % print_freq == 0:
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(train_loader),
                                                                        avg_loss / float(i + 1)))
                wandb.log({'loss/train': avg_loss / float(i + 1)})

    def test_loop(self, test_loader, return_std=False):  # overwrite parrent function
        correct = 0
        count = 0
        acc_all = []

        iter_num = len(test_loader)
        for i, (x, y) in enumerate(test_loader):
            if isinstance(x, list):
                self.n_query = x[0].size(1) - self.n_support
                assert self.n_way == x[0].size(0), "MAML do not support way change"
            else:
                self.n_query = x.size(1) - self.n_support
                assert self.n_way == x.size(0), "MAML do not support way change"

            if self.type == "classification":
                correct_this, count_this = self.correct(x)
                acc_all.append(correct_this / count_this * 100)
            else:
                # Use pearson correlation
                acc_all.append(self.correlation(x, y))

        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)

        if self.type == "classification":
            print('%d Accuracy = %4.2f%% +- %4.2f%%' % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num)))
        else:
            # print correlation
            print('%d Correlation = %4.2f +- %4.2f' % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num)))

        if return_std:
            return acc_mean, acc_std
        else:
            return acc_mean
