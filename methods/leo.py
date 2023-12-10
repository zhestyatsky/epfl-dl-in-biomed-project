import numpy as np
import torch
import torch.nn as nn
import wandb
import math

from backbones.blocks import Linear_fw
from methods.meta_template import MetaTemplate


class NormalDistribution(nn.Module):
    """
    A module for handling Gaussian distributions in the context of LEO algorithm.

    This class provides functionalities for generating output based on Gaussian
    distributions and calculating the Kullback–Leibler divergence (how one
    probability distribution P is different from another reference probability).
    """

    def __init__(self, output_dim):
        """
        Args:
            output_dim (int): The dimensionality of the output distribution.
        """
        super().__init__()
        self.output_dim = output_dim
        # A small offset added to standard deviation to avoid division by zero:
        self.std_offset = 1e-10

        # Pre-initialized means for the Gaussian distribution:
        self.gaussian_means = torch.zeros(self.output_dim)
        # Pre-initialized standard deviations for the Gaussian distribution:
        self.gaussian_stds = torch.ones(self.output_dim)

        if torch.cuda.is_available():
            self.gaussian_means = self.gaussian_means.cuda()
            self.gaussian_stds = self.gaussian_stds.cuda()

    def forward(self, means, stds):
        """
        Generates output based on provided means and standard deviations and calculates the Kullback–Leibler divergence
        """
        stds = nn.functional.softplus(stds)
        gaussian_vector = torch.normal(self.gaussian_means, self.gaussian_stds)

        if torch.cuda.is_available():
            gaussian_vector = gaussian_vector.cuda()

        output = gaussian_vector * stds + means
        kl = self.kl_divergence(output, means, stds)
        return output, kl

    def log_prob(self, x, means, stds):
        """
        Computes log probability density.
        """
        log_prob_density = - 0.5 * ((x - means) / (stds + self.std_offset)) ** 2
        normalization_const = torch.log(stds + self.std_offset) + 0.5 * math.log(2 * math.pi)
        return log_prob_density - normalization_const

    def kl_divergence(self, x, means, stds):
        """
        Calculates the KL divergence for the given inputs.
        """
        kl_components = self.log_prob(x, means, stds) - self.log_prob(x, self.gaussian_means, self.gaussian_stds)
        kl = kl_components.mean()
        return kl


class EncodingNetwork(nn.Module):
    """
    The encoding network component of the LEO algorithm.

    This network encodes the support set inputs and processes them through a relation network
    to produce latent distributions for each class.
    """

    def __init__(self, n_support, n_way, x_dim, encoder_dim, dropout):
        """
        Args:
            n_support (int): Number of support samples per class.
            n_way (int): Number of classes.
            x_dim (int): Dimensionality of the encoder's input.
            encoder_dim (int): Dimensionality of the encoder's output.
            dropout (nn.Module): Dropout for regularization.
        """
        super().__init__()
        self.n_support = n_support
        self.n_way = n_way
        self.encoder_dim = encoder_dim
        self.dropout = dropout

        self.encoding_layer = nn.Linear(x_dim, encoder_dim, bias=False)
        # The relation network takes pairs of encoded inputs to be able to compare them,
        # and it outputs transformed features that encode the relationship between the pair
        self.relation_net = nn.Sequential(
            nn.Linear(2 * encoder_dim, 2 * encoder_dim, bias=False),
            nn.ReLU(),
            nn.Linear(2 * encoder_dim, 2 * encoder_dim, bias=False),
            nn.ReLU(),
            nn.Linear(2 * encoder_dim, 2 * encoder_dim, bias=False),
        )
        self.normal_distribution = NormalDistribution(output_dim=self.n_way * self.encoder_dim)

    def forward(self, x_support):
        """
        Processes the support set through the network, generates output and
        calculates the Kullback–Leibler divergence.

        Args:
            x_support (tensor): (n-ways x n-shots) x M features where shots of the same class are contiguous
        """
        x_support = self.dropout(x_support)
        encoded_x_support = self.encoding_layer(x_support)

        # Form pairs of all encoded inputs to compare each one with everybody else
        # - left-hand: (n-ways x n-shots) x M features
        #   => (n-ways x n-shots) x 1 x M features
        #   => (n-ways x n-shots) x (n-ways x n-shots) x H features
        lhs_relation_net_input = encoded_x_support.unsqueeze(1).tile((1, self.n_way * self.n_support, 1))
        # - right-hand: (n-ways x n-shots) x M features
        #   => 1 x (n-ways x n-shots) x M features
        #   => (n-ways x n-shots) x (n-ways x n-shots) x H features
        rhs_relation_net_input = encoded_x_support.unsqueeze(0).tile((self.n_way * self.n_support, 1, 1))
        # - Concatenate the left-hand and right-hand inputs to form the complete input to the relation network:
        #   => (n-ways x n-shots) x (n-ways x n-shots) x (2 x H features)
        relation_net_input = torch.cat((lhs_relation_net_input, rhs_relation_net_input), dim=-1)

        # Pass pairs through relation network
        relation_net_output = self.relation_net(relation_net_input)

        # Aggregate the pairwise comparisons
        # - (n-ways x n-shots) x (n-ways x n-shots) x (2 x H features)
        #   => (n-ways x n-shots) x (2 x H features)
        #   => n-ways x n-shots x (2 x H features)
        #   => n-ways x (2 x H features)
        relation_net_per_class_output = relation_net_output.mean(dim=1)
        relation_net_per_class_output = relation_net_per_class_output.view(self.n_way, self.n_support, -1)
        relation_net_per_class_output = relation_net_per_class_output.mean(dim=1)

        # Alternative equivalent code:
        # relation_net_per_class_output = relation_net_output.view(self.n_way, self.n_way*self.n_support*self.n_support, -1)
        # relation_net_per_class_output = torch.mean(relation_net_per_class_output, dim=1)

        # The encoded features are effectively doubled to encode both means and
        # standard deviations for each class, which now we split:
        means, stds = relation_net_per_class_output.chunk(chunks=2, dim=-1)
        # Go form 2D (n-way x H features) to 1D:
        means = means.contiguous().view(self.n_way * self.encoder_dim)
        stds = stds.contiguous().view(self.n_way * self.encoder_dim)

        # Generate final output n-way x H features
        output, kl_div = self.normal_distribution(means, stds)
        output = output.view(self.n_way, self.encoder_dim)

        return output, kl_div


class DecodingNetwork(nn.Module):
    """
    The decoding network component of the LEO algorithm.

    This network decodes the latent outputs from the encoding network to produce
    the final task-specific parameters.
    """

    def __init__(self, n_way, latent_dim, output_dim):
        """
        Args:
            n_way (int): Number of classes.
            latent_dim (int): Dimensionality of the latent space.
            output_dim (int): Dimensionality of the final output.
        """
        super().__init__()
        self.n_way = n_way
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        # The decoding layer is designed to produce an output that is twice the
        # size of the output_dim. This is because the output is intended to
        # represent both the means and standard deviations for a Gaussian
        # distribution:
        self.decoding_layer = nn.Linear(self.latent_dim, 2 * output_dim, bias=False)
        self.normal_distribution = NormalDistribution(output_dim=self.n_way * output_dim)

    def forward(self, latent_output):
        """
        Decodes the latent output to produce final parameters.

        Args:
            latent_output (tensor): n-ways x H features
        """
        # Decode:
        # - (n-ways x H features) => (2 x output dimension)
        decoded_output = self.decoding_layer(latent_output)
        # Splits the decoded output into means and standard deviations:
        means, stds = decoded_output.chunk(chunks=2, dim=-1)
        means = means.contiguous().view(self.n_way * self.output_dim)
        stds = stds.contiguous().view(self.n_way * self.output_dim)
        # Generates samples from a Normal distribution using the above parameters
        # - (n_way x output dimension)
        output, _ = self.normal_distribution(means, stds)
        output = output.view(self.n_way, self.output_dim)
        return output


class LEO(MetaTemplate):
    """
    Latent Embedding Optimization (LEO) algorithm for few-shot learning.

    This class integrates the encoding and decoding networks and provides methods
    for training and testing the LEO model.
    """

    def __init__(self, x_dim, backbone_dims, backbone, n_way, n_support, n_task, inner_lr_init, finetuning_lr_init,
                 num_adaptation_steps, kl_coef, orthogonality_penalty_coef, encoder_penalty_coef, dropout,
                 gradient_threshold, gradient_norm_threshold, latent_space_dim, optimize_backbone, do_pretrain_weights):
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
            num_adaptation_steps (int): Number of inner and fine-tuning loops adaptation steps.
            kl_coef (float): Coefficient for KL divergence penalty.
            orthogonality_penalty_coef (float): Coefficient for orthogonality penalty.
            encoder_penalty_coef (float): Coefficient for encoder penalty.
            dropout (float): Dropout probability.
            gradient_threshold (float): Threshold for gradient clipping.
            gradient_norm_threshold (float): Threshold for gradient norm clipping.
            latent_space_dim (int): Dimensionality of the latent space.
            optimize_backbone (bool): If True then both classifier and backbone weights are optimized.
            do_pretrain_weights (bool): Pretrain backbone and classifier weights without metalearning.
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
        self.num_adaptation_steps = num_adaptation_steps
        self.kl_coef = kl_coef
        self.orthogonality_penalty_coef = orthogonality_penalty_coef
        self.encoder_penalty_coef = encoder_penalty_coef
        self.gradient_threshold = gradient_threshold
        self.gradient_norm_threshold = gradient_norm_threshold
        self.optimize_backbone = optimize_backbone
        self.do_pretrain_weights = do_pretrain_weights

        assert not self.optimize_backbone, (
            "Backbone optimization is not supported. Only classifier weights are optimized"
        )

        self.dropout = nn.Dropout(p=dropout)
        self.encoder = EncodingNetwork(
            n_support=n_support, n_way=n_way, x_dim=x_dim, encoder_dim=self.latent_space_dim, dropout=self.dropout,
        )

        self.decoder = DecodingNetwork(
            n_way=self.n_way, latent_dim=self.latent_space_dim, output_dim=self.feat_dim + 1,
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
        clf_weight, clf_bias = weights.split([self.feat_dim, 1], dim=-1)
        clf_bias = clf_bias.squeeze()
        self.classifier.weight.fast = clf_weight
        self.classifier.bias.fast = clf_bias

    def calculate_scores_and_regularization_parameters(self, x, y=None):
        """
        Args:
            x (tensor): n-shots x (k-shots + query shots) x M features
        """
        if torch.cuda.is_available():
            x = x.cuda()

        # For the current episode, build the support and query datasets (for training and validation, respectively)
        # in 2D instead of 3D:
        #   n-shots x (k-shots + query shots) x M features
        #       => (n-shots x k-shots) x M features
        #       => (n-shots x query shots) x M features
        # and shots of the same class are kept contiguous
        x_support = x[:, :self.n_support, :].contiguous().view(self.n_way * self.n_support, -1)
        x_query = x[:, self.n_support:, :].contiguous().view(self.n_way * self.n_query, -1)

        if y is None:  # Classification task, assign labels (class indices) based on n_way
            y_support = torch.from_numpy(np.repeat(range(self.n_way), self.n_support))
        else:  # Regression task, keep labels as they are
            y_support = y[:, :self.n_support].contiguous().view(self.n_way * self.n_support, -1)

        if torch.cuda.is_available():
            y_support = y_support.cuda()

        self.zero_grad()

        if self.do_pretrain_weights:
            return self.forward(x)

        # Initialize classifier (and, optionally, backbone) conditioned to the support dataset
        latents_z, kl_div = self.encoder(x_support)
        latents_z_init = latents_z.detach()
        weights = self.decoder(latents_z)
        self.set_weights(weights)

        # Meta training inner loop (aka the adaptation procedure)
        # - updates the classiffier weights by decoding modified latent z values (modified using the computed gradient)
        for _ in range(self.num_adaptation_steps):
            scores = self.forward(x_support)
            set_loss = self.loss_fn(scores, y_support)
            # PyTorch keeps track of all operations that affect tensors, which allows it to automatically calculate derivatives
            # such as to compute the gradient of the loss function (set_loss) with respect to the latent variables (latents_z):
            grad = torch.autograd.grad(set_loss, latents_z, create_graph=True)[0]
            latents_z = latents_z - self.inner_lr * grad
            weights = self.decoder(latents_z)
            self.set_weights(weights)

        # Meta training fine-tuning loop
        # - updates the classiffier weights by using the computed gradient
        for _ in range(self.num_adaptation_steps):
            scores = self.forward(x_support)
            set_loss = self.loss_fn(scores, y_support)
            grad = torch.autograd.grad(set_loss, weights, create_graph=True)[0]
            weights = weights - self.finetuning_lr * grad
            self.set_weights(weights)

        scores = self.forward(x_query)
        encoder_penalty = torch.mean((latents_z_init - latents_z) ** 2)

        return scores, kl_div, encoder_penalty

    def set_forward(self, x, y=None):
        if self.do_pretrain_weights:
            return self.calculate_scores_and_regularization_parameters(x, y)
        scores, kl_div, encoder_penalty = self.calculate_scores_and_regularization_parameters(x, y)
        return scores

    def set_forward_adaptation(self, x, is_feature=False):  # overwrite parrent function
        raise ValueError('MAML performs further adapation simply by increasing task_upate_num')

    def set_forward_loss(self, x, y=None):
        if y is None:  # Classification task
            y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
        else:  # Regression task
            y_query = y[:, self.n_support:].contiguous().view(self.n_way * self.n_query, -1)

        if torch.cuda.is_available():
            y_query = y_query.cuda()

        if self.do_pretrain_weights:
            scores = self.calculate_scores_and_regularization_parameters(x, y)
            return self.loss_fn(scores, y_query)

        scores, kl_div, encoder_penalty = self.calculate_scores_and_regularization_parameters(x, y)

        decoder_parameters = list(self.decoder.parameters())
        orthogonality_penalty = self.orthogonality(decoder_parameters[0])

        loss = self.loss_fn(scores, y_query)
        regularized_loss = (
                loss +
                self.kl_coef * kl_div +
                self.encoder_penalty_coef * encoder_penalty +
                self.orthogonality_penalty_coef * orthogonality_penalty
        )

        return regularized_loss

    def train_loop(self, epoch, train_loader, optimizer):  # overwrite parrent function
        """
        Training loop for the LEO model
        """
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
                if not self.do_pretrain_weights:
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
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(train_loader), avg_loss / float(i + 1)))
                print('InnerLR {:f}'.format(self.inner_lr.item()))
                print('FineTuningLR {:f}'.format(self.finetuning_lr.item()))
                wandb.log({'loss/train': avg_loss / float(i + 1)})

    def test_loop(self, test_loader, return_std=False):  # overwrite parrent function
        """
        Testing loop for evaluating the LEO model
        """
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
