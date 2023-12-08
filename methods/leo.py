import numpy as np
import torch
import torch.nn as nn
import wandb
import math

from backbones.blocks import Linear_fw
from methods.meta_template import MetaTemplate


class NormalDistribution(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.output_dim = output_dim
        self.std_offset = 1e-10

        self.gaussian_means = torch.zeros(self.output_dim)
        self.gaussian_stds = torch.ones(self.output_dim)

        if torch.cuda.is_available():
            self.gaussian_means = self.gaussian_means.cuda()
            self.gaussian_stds = self.gaussian_stds.cuda()

    def forward(self, means, std_logs):
        stds = torch.exp(std_logs / 2)
        gaussian_vector = torch.normal(self.gaussian_means, self.gaussian_stds)

        if torch.cuda.is_available():
            gaussian_vector = gaussian_vector.cuda()

        output = gaussian_vector * stds + means
        kl = self.kl_divergence(output, means, stds)
        return output, kl

    def log_prob(self, x, means, stds):
        log_prob_density = - 0.5 * ((x - means) / (stds + self.std_offset)) ** 2
        normalization_const = torch.log(stds + self.std_offset) + 0.5 * math.log(2 * math.pi)
        return log_prob_density - normalization_const

    def kl_divergence(self, x, means, stds):
        kl_components = self.log_prob(x, means, stds) - self.log_prob(x, self.gaussian_means, self.gaussian_stds)
        kl = kl_components.mean()
        return kl


class EncodingNetwork(nn.Module):
    def __init__(self, n_support, n_way, x_dim, encoder_dim, dropout):
        super().__init__()
        self.n_support = n_support
        self.n_way = n_way
        self.encoder_dim = encoder_dim
        self.dropout = dropout

        self.encoding_layer = nn.Linear(x_dim, encoder_dim)
        self.relation_net = nn.Linear(2 * encoder_dim, 2 * encoder_dim)
        self.normal_distribution = NormalDistribution(output_dim=self.n_way*self.encoder_dim)

    def forward(self, x_support):
        x_support = self.dropout(x_support)
        encoded_x_support = self.encoding_layer(x_support)

        lhs_relation_net_input = encoded_x_support.unsqueeze(1).tile((1, self.n_way * self.n_support, 1))
        rhs_relation_net_input = encoded_x_support.unsqueeze(0).tile((self.n_way * self.n_support, 1, 1))
        relation_net_input = torch.cat((lhs_relation_net_input, rhs_relation_net_input), dim=-1)

        relation_net_output = self.relation_net(relation_net_input).mean(dim=1)
        relation_net_per_class_output = relation_net_output.view(self.n_way, self.n_support, -1).mean(dim=1)

        means, std_logs = relation_net_per_class_output.chunk(chunks=2, dim=-1)
        means = means.contiguous().view(self.n_way * self.encoder_dim)
        std_logs = std_logs.contiguous().view(self.n_way * self.encoder_dim)
        output, kl_div = self.normal_distribution(means, std_logs)
        output = output.view(self.n_way, self.encoder_dim)
        return output, kl_div


class DecodingNetwork(nn.Module):
    def __init__(self, n_way, latent_dim, output_dim):
        super().__init__()
        self.n_way = n_way
        self.latent_dim = latent_dim
        self.decoding_layer = nn.Linear(self.n_way*self.latent_dim, 2 * output_dim)
        self.normal_distribution = NormalDistribution(output_dim=output_dim)

    def forward(self, latent_output):
        latent_output = latent_output.view(self.n_way*self.latent_dim)
        decoded_output = self.decoding_layer(latent_output)
        means, std_logs = decoded_output.chunk(chunks=2, dim=-1)
        output, _ = self.normal_distribution(means, std_logs)
        return output


class LEO(MetaTemplate):
    def __init__(self, x_dim, backbone_dims, backbone, n_way, n_support, n_task, inner_lr_init, finetuning_lr_init,
                 num_inner_steps, num_finetuning_steps, kl_coef, orthogonality_penalty_coef, 
                 encoder_penalty_coef, dropout, gradient_threshold, gradient_norm_threshold, latent_space_dim):
        """
            Initialize the LEO (Latent Embedding Optimization) model.

            Args:
                x_dim (int): Input data dimension.
                backbone_dims (List[int]): Backbone layer dimensions.
                backbone (object): The backbone of the model.
                n_way (int): Number of classes in each task.
                n_support (int): Number of support examples per class.
                n_task (int): Number of tasks.
                inner_lr_init (float): Initial inner loop learning rate.
                finetuning_lr_init (float): Initial finetuning loop learning rate.
                num_inner_steps (int): Number of inner loop adaptation steps.
                num_finetuning_steps (int): Number of inner loop finetuning steps.
                kl_coef (float): Coefficient for KL divergence penalty.
                orthogonality_penalty_coef (float): Coefficient for orthogonality penalty.
                encoder_penalty_coef (float): Coefficient for encoder penalty.
                dropout (float): Dropout probability.
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

        self.backbone_dims = backbone_dims
        self.n_task = n_task

        # LEO specific hyperparameters
        self.latent_space_dim = latent_space_dim
        self.inner_lr_init = inner_lr_init
        self.finetuning_lr_init = finetuning_lr_init
        self.num_inner_steps = num_inner_steps
        self.num_finetuning_steps = num_finetuning_steps
        self.kl_coef = kl_coef
        self.orthogonality_penalty_coef = orthogonality_penalty_coef
        self.encoder_penalty_coef = encoder_penalty_coef
        self.gradient_threshold = gradient_threshold
        self.gradient_norm_threshold = gradient_norm_threshold

        self.dropout = nn.Dropout(p=dropout)
        self.encoder = EncodingNetwork(
            n_support=n_support, n_way=n_way, x_dim=x_dim, encoder_dim=self.latent_space_dim, dropout=self.dropout,
        )

        # We add +1 because of the classifier bias vector
        self.weights_matrices_dims = [(self.feat_dim + 1,  n_way)]
        for layer_idx in range(len(backbone_dims)):
            # We add +3 because each backbone block has bias vector in the Linear layer and 2 vectors in the BatchNorm
            if layer_idx == 0:
                self.weights_matrices_dims.append((x_dim + 3, backbone_dims[layer_idx]))
            else:
                self.weights_matrices_dims.append((backbone_dims[layer_idx-1] + 3, backbone_dims[layer_idx]))

        self.decoder = DecodingNetwork(
            n_way=self.n_way,
            latent_dim=self.latent_space_dim,
            output_dim=sum(dims[0] * dims[1] for dims in self.weights_matrices_dims),
        )

        self.inner_lr = nn.Parameter(torch.tensor(inner_lr_init, dtype=torch.float32))
        self.finetuning_lr = nn.Parameter(torch.tensor(finetuning_lr_init, dtype=torch.float32))

    def forward(self, x):
        out = self.feature.forward(x)
        scores = self.classifier.forward(out)

        # For regression tasks, these are not scores but predictions
        if scores.shape[1] == 1:
            scores = scores.squeeze(1)

        return scores

    def orthogonality(self, weight):
        """
            Calculate the orthogonality penalty for a weight matrix.

            Args:
            - weight (torch.Tensor): The weight matrix to calculate the orthogonality penalty for.

            Returns:
            - torch.Tensor: The orthogonality penalty, computed as the mean squared difference
            between the correlation matrix of the weight matrix and the identity matrix.
        """
        w_square = weight @ weight.t()
        w_norm = torch.norm(weight, dim=1, keepdim=True) + 1e-10
        correlation_matrix = w_square / (w_norm @ w_norm.t())
        identity = torch.eye(correlation_matrix.size(0))
        if torch.cuda.is_available():
            identity = identity.cuda()
        return torch.mean((correlation_matrix - identity) ** 2)

    def set_weights(self, weights):
        weights_vectors_dims = [dim[0] * dim[1] for dim in self.weights_matrices_dims]
        weights_components = weights.split(weights_vectors_dims)

        clf_weights = weights_components[0]
        backbone_blocks_weights = weights_components[1:]

        clf_matrix_dims = self.weights_matrices_dims[0]
        backbone_matrices_dims = self.weights_matrices_dims[1:]

        # First we set the classifier weights
        clf_weight, clf_bias = clf_weights.view(clf_matrix_dims).split([clf_matrix_dims[0] - 1, 1])
        clf_bias = clf_bias.squeeze()
        self.classifier.weight.fast = clf_weight.T
        self.classifier.bias.fast = clf_bias

        # Then we set the backbone weights
        for layer_idx, block_weights in enumerate(backbone_blocks_weights):
            block_matrix_dims = backbone_matrices_dims[layer_idx]
            # Each backbone block contains a linear layer with a bias followed by a BatchNorm
            lin_weight, lin_bias, batch_norm_weight, batch_norm_bias = (
                block_weights.view(backbone_matrices_dims[layer_idx]).split([block_matrix_dims[0] - 3, 1, 1, 1])
            )
            lin_bias = lin_bias.squeeze()
            batch_norm_weight = batch_norm_weight.squeeze()
            batch_norm_bias = batch_norm_bias.squeeze()
            self.feature.encoder[layer_idx][0].weight.fast = lin_weight.T
            self.feature.encoder[layer_idx][0].bias.fast = lin_bias
            self.feature.encoder[layer_idx][1].weight.fast = batch_norm_weight
            self.feature.encoder[layer_idx][1].bias.fast = batch_norm_bias

    def calculate_scores_and_regularization_parameters(self, x, y=None):
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

        latents_z, kl_div = self.encoder(x_support)
        latents_z_init = latents_z.detach()
        weights = self.decoder(latents_z)
        self.set_weights(weights)

        # Meta training inner loop
        for i in range(self.num_inner_steps):
            scores = self.forward(x_support)
            set_loss = self.loss_fn(scores, y_support)
            grad = torch.autograd.grad(set_loss, latents_z, create_graph=True)[0]
            latents_z = latents_z - self.inner_lr * grad
            weights = self.decoder(latents_z)
            self.set_weights(weights)

        # Meta training fine-tuning loop
        for i in range(self.num_finetuning_steps):
            scores = self.forward(x_support)
            set_loss = self.loss_fn(scores, y_support)
            grad = torch.autograd.grad(set_loss, weights, create_graph=True)[0]
            weights = weights - self.finetuning_lr * grad
            self.set_weights(weights)

        scores = self.forward(x_query)
        encoder_penalty = torch.mean((latents_z_init - latents_z) ** 2)

        return scores, kl_div, encoder_penalty

    def set_forward(self, x, y=None):
        scores, kl_div, encoder_penalty = self.calculate_scores_and_regularization_parameters(x, y)
        return scores

    def set_forward_adaptation(self, x, is_feature=False):  # overwrite parrent function
        raise ValueError('MAML performs further adapation simply by increasing task_upate_num')

    def set_forward_loss(self, x, y=None):
        scores, kl_div, encoder_penalty = self.calculate_scores_and_regularization_parameters(x, y)

        # TODO: Perhaps include decoder bias
        #orthogonality_penalty = self.orthogonality(list(self.decoder.parameters())[0])

        if y is None:  # Classification task
            y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
        else:  # Regression task
            y_query = y[:, self.n_support:].contiguous().view(self.n_way * self.n_query, -1)

        if torch.cuda.is_available():
            y_query = y_query.cuda()

        loss = self.loss_fn(scores, y_query)
        regularized_loss = (
                loss +
                self.kl_coef * kl_div +
                self.encoder_penalty_coef * encoder_penalty# +
                #self.orthogonality_penalty_coef * orthogonality_penalty
        )

        return regularized_loss

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

                # Check for NaN values in the gradients
                for param in [*self.encoder.parameters(), *self.decoder.parameters(), self.inner_lr, self.finetuning_lr]:
                    if torch.isnan(param.grad).any():
                        # Create a mask for NaN values in the gradients
                        nan_mask = torch.isnan(param.grad)
                        # Zero out the gradients for parameters associated with NaN values
                        param.grad[nan_mask] = 0.0

                # clip gradient if necessary
                nn.utils.clip_grad_value_([*self.encoder.parameters(), *self.decoder.parameters(), self.inner_lr, self.finetuning_lr], self.gradient_threshold)
                nn.utils.clip_grad_norm_([*self.encoder.parameters(), *self.decoder.parameters(), self.inner_lr, self.finetuning_lr], self.gradient_norm_threshold)

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
