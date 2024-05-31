
from torch import optim, nn, utils, Tensor
import lightning.pytorch as pl
import torch
import numpy as np
import time
import os
from lightning.pytorch import loggers as pl_loggers
import shutil

class MapStyleDataset(object):

    def __init__(self, *args, device='cpu', dataset='train', train_size=0.8, seed=0):

        n_samples = args[0].shape[0]
        self.rng = np.random.default_rng(seed=seed)
        original_indices = self.rng.permutation(n_samples)
        n_train = int(train_size * n_samples)
        itr = original_indices[:n_train] # train set
        iva = original_indices[n_train:] # val set

        if dataset == 'train':
            original_indices = itr
        elif dataset == 'val':
            original_indices = iva

        self.original_indices = original_indices


        # Subsample data and convert to torch.Tensor with the right device
        self.args = []
        for arg in args:
            arg = arg[original_indices]
            arg = torch.from_numpy(arg).to(device)
            self.args.append(arg)

    def __getitem__(self, index):
        res = []
        for arg in self.args:
            res.append(arg[index])
        return (*res,)

    def __len__(self):
        return len(self.original_indices)


ACTIVATIONS = {
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
    "elu": nn.ELU,
    "leakyrelu": nn.LeakyReLU,
    "tanh": nn.Tanh
}

import logging

logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.WARNING)
logging.getLogger("lightning.pytorch.accelerators.cuda").setLevel(logging.WARNING)
logging.getLogger("lightning.fabric.utilities.seed").setLevel(logging.WARNING)
logging.getLogger("lightning.pytorch.utilities.seed").setLevel(logging.WARNING)

def shape_one(arg, regression=True):
    if len(arg.shape) == 1 or arg.shape[1] == 1:
        output = 1 if regression else int(np.max(arg) + 1)
        return output
    else:
        return arg.shape[1]

def neural_network_fitter(*args, model_class=None, regression=True, remove_logs=False, **kwargs):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    assert not(kwargs.get('early_stopping',True) and not kwargs.get('validation', kwargs.get('early_stopping',True)))
    train_data = MapStyleDataset(*args, device=device, dataset='train' if kwargs.get('validation', True) else 'all')
    val_data = MapStyleDataset(*args, device=device, dataset='val')
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=kwargs.get('batch_size',64), shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=kwargs.get('batch_size',64), shuffle=True)
    model_kwargs = {
        key: value for (key,value) in kwargs.items() if key in ['x_dim', 'a_dim', 'init_lr', 'balancing_x_dim', 'layer_dims_balancing_x', 'layer_dims_output', 'dropout_prob', 'weight_decay', 'optimizer', 'activation', 'final_activation_output']
    }
    model = model_class(*[shape_one(arg, regression=regression) for arg in args], **model_kwargs)
    pl.seed_everything(kwargs.get('random_state',0))
    early_stop_callback = pl.callbacks.early_stopping.EarlyStopping(
        monitor='val_loss',
        min_delta=kwargs.get('min_delta',0),
        patience=kwargs.get('patience',3),
        verbose=False,
        mode='min'
    )
    log_dir = f"../outputs/logs/{time.time()}/"
    os.makedirs(log_dir + 'lightning_logs/')
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=log_dir)
    if kwargs.get('early_stopping',True):
        trainer = pl.Trainer(deterministic=True, logger=tb_logger, check_val_every_n_epoch=kwargs.get('check_val_every_n_epoch',1),
                             max_epochs=kwargs.get('n_epochs',100), callbacks=[early_stop_callback],
                             enable_progress_bar=False, enable_model_summary=False)
        trainer.validate(model, train_loader, verbose=False)
        trainer.validate(model, val_loader, verbose=False)
        model.train()
        trainer.fit(model, train_loader, val_loader)
    else:
        trainer = pl.Trainer(deterministic=True, logger=tb_logger,
                             check_val_every_n_epoch=kwargs.get('check_val_every_n_epoch', 1),
                             max_epochs=kwargs.get('n_epochs', 100),
                             enable_progress_bar=False, enable_model_summary=False)
        model.train()
        trainer.fit(model, train_loader)
    if remove_logs:
        shutil.rmtree(log_dir)
    model.eval()
    return model


def build_fc_network(layer_dims, activation="relu", dropout_prob=0., final_activation=True):
    """
    Stacks multiple fully-connected layers with an activation function and a dropout layer in between.

    - Source used as orientation: https://github.com/eelxpeng/UnsupervisedDeepLearning-Pytorch/blob/master/udlp/clustering/vade.py

    Args:
        layer_dims: A list of integers, where (starting from 1) the (i-1)th and ith entry indicates the input
                    and output dimension of the ith layer, respectively.
        activation: Activation function to choose.
        dropout_prob: Dropout probability between every fully connected layer with activation.

    Returns:
        An nn.Sequential object of the layers.
    """
    # Note: possible alternative: OrderedDictionary
    net = []
    for i in range(1, len(layer_dims)):
        net.append(nn.Linear(layer_dims[i-1], layer_dims[i]).double())
        if i < len(layer_dims) - 1 or final_activation:
            net.append(ACTIVATIONS[activation]())
        net.append(nn.Dropout(dropout_prob))
    net = nn.Sequential(*net)  # unpacks list as separate arguments to be passed to function

    return net

OPTIMIZERS = {
        "adam": optim.Adam,
        "sgd": optim.SGD,
    }




class BalancingScoreRieszNetPQ(pl.LightningModule):
    def __init__(self, x_dim, labels_dim=None, init_lr=0.01, balancing_x_dim=5, layer_dims_balancing_x=[], layer_dims_output=[200,200,200], dropout_prob=0., weight_decay=0., optimizer="adam", activation="relu", final_activation_output='exp', **kwargs):
        super().__init__()

        self.final_activation_output = final_activation_output
        final_activation_output_bool = final_activation_output if isinstance(final_activation_output, bool) else False

        final_layer_x = [] if balancing_x_dim is None else [balancing_x_dim]
        self.score_x = build_fc_network([x_dim] + layer_dims_balancing_x + final_layer_x, dropout_prob=dropout_prob, activation=activation, final_activation=False)
        self.intermediary_dim = ([x_dim] + layer_dims_balancing_x + final_layer_x)[-1]

        self.output_net = build_fc_network([self.intermediary_dim] + layer_dims_output + [1], dropout_prob=dropout_prob, activation=activation, final_activation=final_activation_output_bool)

        self.activation = activation
        self.optimizer = optimizer
        self.init_lr = init_lr
        self.weight_decay = weight_decay

        for kwarg in kwargs:
            print(f'WARNING : {kwarg} not used by BalancingScoreNet.')

    def riesz(self, x):
        x_rep = ACTIVATIONS[self.activation]()(self.score_x(x))

        prediction = self.output_net(x_rep)
        if isinstance(self.final_activation_output, str) and self.final_activation_output == 'exp':
            prediction = torch.exp(prediction)
        return prediction



    def training_step(self, batch, batch_idx, name='train_loss'):
        x, labels = batch
        xp = x[torch.flatten(labels) == 0]
        xq = x[torch.flatten(labels) == 1]
        prediction_p = self.riesz(xp)
        prediction_q = self.riesz(xq)
        loss = torch.mean(torch.pow(prediction_p, 2)) - 2*torch.mean(prediction_q)
        self.log(name, loss)
        return loss

    def validation_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx, name='val_loss')

    def configure_optimizers(self):
        # define optimizer and scheduler
        optimizer = OPTIMIZERS[self.optimizer](
            filter(lambda p: p.requires_grad, self.parameters()), lr=self.init_lr,weight_decay=self.weight_decay
        )

        # training and evaluation loop
        epoch_lr = optimizer.param_groups[0]['lr']

        # TODO add support for learning rate scheduler
        # adjust learning rate
        # if epoch % args.update_lr_every_epoch == 0 and not epoch == 0:
        #     adjust_learning_rate(optimizer, epoch_lr, args.lr_decay, args.min_lr)

        return optimizer


class BalancingScoreRieszNetDifferentOutputs(pl.LightningModule):
    def __init__(self, x_dim, a_dim, init_lr=0.01, balancing_x_dim=5, layer_dims_balancing_x=[], layer_dims_output=[200,200,200], dropout_prob=0., weight_decay=0., optimizer="adam", activation="relu", final_activation_output='exp', **kwargs):
        super().__init__()

        self.final_activation_output = final_activation_output
        final_activation_output_bool = final_activation_output if isinstance(final_activation_output, bool) else False

        final_layer_x = [] if balancing_x_dim is None else [balancing_x_dim]
        self.score_x = build_fc_network([x_dim] + layer_dims_balancing_x + final_layer_x, dropout_prob=dropout_prob, activation=activation, final_activation=False)
        self.intermediary_dim = ([x_dim] + layer_dims_balancing_x + final_layer_x)[-1]

        self.output_net = nn.ModuleList([
            build_fc_network([self.intermediary_dim] + layer_dims_output + [1], dropout_prob=dropout_prob, activation=activation, final_activation=final_activation_output_bool) for _ in range(a_dim)
        ])

        self.activation = activation
        self.optimizer = optimizer
        self.init_lr = init_lr
        self.weight_decay = weight_decay

        for kwarg in kwargs:
            print(f'WARNING : {kwarg} not used by BalancingScoreNet.')

    def riesz_all(self, x, a):
        x_rep = ACTIVATIONS[self.activation]()(self.score_x(x))

        prediction = torch.cat([
            torch.reshape(output(x_rep), (-1, 1)) for output in self.output_net
        ], dim=-1)
        if isinstance(self.final_activation_output, str) and self.final_activation_output == 'exp':
            prediction = torch.exp(prediction)
        return prediction

    def riesz(self, x, a):
        prediction = self.riesz_all(x, a)
        prediction_pa = torch.sum(prediction * a, dim=-1)
        return prediction_pa



    def training_step(self, batch, batch_idx, name='train_loss'):
        x, a = batch
        pas = torch.mean(a, axis=0)
        prediction = self.riesz_all(x, a)
        prediction_pa = torch.sum(prediction * a, dim=-1)
        prediction_p = torch.sum(prediction * pas, dim=-1)
        loss = torch.sum(torch.pow(prediction_pa, 2) - 2*prediction_p)
        self.log(name, loss)
        return loss

    def validation_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx, name='val_loss')

    def configure_optimizers(self):
        # define optimizer and scheduler
        optimizer = OPTIMIZERS[self.optimizer](
            filter(lambda p: p.requires_grad, self.parameters()), lr=self.init_lr,weight_decay=self.weight_decay
        )

        # training and evaluation loop
        epoch_lr = optimizer.param_groups[0]['lr']

        # TODO add support for learning rate scheduler
        # adjust learning rate
        # if epoch % args.update_lr_every_epoch == 0 and not epoch == 0:
        #     adjust_learning_rate(optimizer, epoch_lr, args.lr_decay, args.min_lr)

        return optimizer



class ClassificationNet(pl.LightningModule):
    def __init__(self, z_dim, labels_dim, init_lr=0.01, layer_dims=[200,200,200], dropout_prob=0., weight_decay=0., optimizer="adam", activation="relu", **kwargs):
        super().__init__()

        self.presoftmax = build_fc_network([z_dim] + layer_dims + [labels_dim], dropout_prob=dropout_prob, activation=activation, final_activation=False)

        self.softmax = nn.Softmax()
        self.loss_fn = nn.CrossEntropyLoss()

        self.activation = activation
        self.optimizer = optimizer
        self.init_lr = init_lr
        self.weight_decay = weight_decay

        for kwarg in kwargs:
            print(f'WARNING : {kwarg} not used by ClassificationNet.')

    def predict_proba(self, z):
        return self.softmax(self.presoftmax(z))

    def training_step(self, batch, batch_idx, name='train_loss'):
        z, labels = batch
        probas = self.predict_proba(z)
        loss =  self.loss_fn(probas, labels)

        self.log(name, loss)
        return loss

    def validation_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx, name='val_loss')

    def configure_optimizers(self):
        # define optimizer and scheduler
        optimizer = OPTIMIZERS[self.optimizer](
            filter(lambda p: p.requires_grad, self.parameters()), lr=self.init_lr,weight_decay=self.weight_decay
        )

        # training and evaluation loop
        epoch_lr = optimizer.param_groups[0]['lr']

        # TODO add support for learning rate scheduler
        # adjust learning rate
        # if epoch % args.update_lr_every_epoch == 0 and not epoch == 0:
        #     adjust_learning_rate(optimizer, epoch_lr, args.lr_decay, args.min_lr)

        return optimizer


class NSMBalancingScoreNet(pl.LightningModule):
    def __init__(self, x_dim, a_dim, init_lr=0.01, balancing_x_dim=5, layer_dims_balancing_x=[], layer_dims_output=[200,200,200], dropout_prob=0., weight_decay=0., optimizer="adam", activation="relu", **kwargs):
        super().__init__()

        final_layer_x = [] if balancing_x_dim is None else [balancing_x_dim]
        self.score_x = build_fc_network([x_dim] + layer_dims_balancing_x + final_layer_x, dropout_prob=dropout_prob, activation=activation, final_activation=False)
        self.intermediary_dim = ([x_dim] + layer_dims_balancing_x + final_layer_x)[-1]
        self.postreprpresoftmax = build_fc_network([self.intermediary_dim] + layer_dims_output + [a_dim], dropout_prob=dropout_prob, activation=activation, final_activation=False)

        self.activation = activation
        self.optimizer = optimizer
        self.init_lr = init_lr
        self.weight_decay = weight_decay

        self.softmax = nn.Softmax()
        self.loss_fn = nn.CrossEntropyLoss()


        for kwarg in kwargs:
            print(f'WARNING : {kwarg} not used by ClassificationNet.')

    def predict_proba(self, x):
        return self.softmax(self.presoftmax(x))

    def presoftmax(self, x):
        x_rep = ACTIVATIONS[self.activation]()(self.score_x(x))
        return self.postreprpresoftmax(x_rep)

    def training_step(self, batch, batch_idx, name='train_loss'):
        x, labels = batch
        probas = self.predict_proba(x)
        labels = torch.flatten(labels)
        loss = self.loss_fn(probas, labels)

        self.log(name, loss)
        return loss

    def validation_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx, name='val_loss')

    def configure_optimizers(self):
        # define optimizer and scheduler
        optimizer = OPTIMIZERS[self.optimizer](
            filter(lambda p: p.requires_grad, self.parameters()), lr=self.init_lr,weight_decay=self.weight_decay
        )

        # training and evaluation loop
        epoch_lr = optimizer.param_groups[0]['lr']

        # TODO add support for learning rate scheduler
        # adjust learning rate
        # if epoch % args.update_lr_every_epoch == 0 and not epoch == 0:
        #     adjust_learning_rate(optimizer, epoch_lr, args.lr_decay, args.min_lr)

        return optimizer



class BalancingScoreNet(pl.LightningModule):
    def __init__(self, x_dim, a_dim, init_lr=0.01, balancing_x_dim=5, balancing_a_dim=None, layer_dims_balancing_x=[], layer_dims_balancing_a=[], layer_dims_output=[200,200,200], dropout_prob=0., weight_decay=0., optimizer="adam", activation="relu", final_activation_output='exp', **kwargs):
        super().__init__()

        self.final_activation_output = final_activation_output
        final_activation_output_bool = final_activation_output if isinstance(final_activation_output, bool) else False

        final_layer_x = [] if balancing_x_dim is None else [balancing_x_dim]
        self.score_x = build_fc_network([x_dim] + layer_dims_balancing_x + final_layer_x, dropout_prob=dropout_prob, activation=activation, final_activation=False)
        final_layer_a = [] if balancing_a_dim is None else [balancing_a_dim]
        self.score_a = build_fc_network([a_dim] + layer_dims_balancing_a + final_layer_a, dropout_prob=dropout_prob, activation=activation, final_activation=False)
        self.intermediary_dim = ([x_dim] + layer_dims_balancing_x + final_layer_x)[-1] + ([a_dim] + layer_dims_balancing_a + final_layer_a)[-1]
        self.output_net = build_fc_network([self.intermediary_dim] + layer_dims_output + [1], dropout_prob=dropout_prob, activation=activation, final_activation=final_activation_output_bool)

        self.activation = activation
        self.optimizer = optimizer
        self.init_lr = init_lr
        self.weight_decay = weight_decay

        for kwarg in kwargs:
            print(f'WARNING : {kwarg} not used by BalancingScoreNet.')

    def training_step(self, batch, batch_idx, name='train_loss'):
        x, a, groundtruth = batch
        x_rep = ACTIVATIONS[self.activation]()(self.score_x(x))
        a_rep = ACTIVATIONS[self.activation]()(self.score_a(a))
        z = torch.cat([x_rep, a_rep], dim=-1)
        prediction = torch.flatten(self.output_net(z))
        if isinstance(self.final_activation_output, str) and self.final_activation_output == 'exp':
            prediction = torch.exp(prediction)
        groundtruth = torch.flatten(groundtruth)
        loss = nn.functional.mse_loss(groundtruth, prediction)
        self.log(name, loss)
        return loss

    def validation_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx, name='val_loss')

    def configure_optimizers(self):
        # define optimizer and scheduler
        optimizer = OPTIMIZERS[self.optimizer](
            filter(lambda p: p.requires_grad, self.parameters()), lr=self.init_lr,weight_decay=self.weight_decay
        )

        # training and evaluation loop
        epoch_lr = optimizer.param_groups[0]['lr']

        # TODO add support for learning rate scheduler
        # adjust learning rate
        # if epoch % args.update_lr_every_epoch == 0 and not epoch == 0:
        #     adjust_learning_rate(optimizer, epoch_lr, args.lr_decay, args.min_lr)

        return optimizer



class BalancingScoreNetDifferentOutputs(pl.LightningModule):
    def __init__(self, x_dim, a_dim, init_lr=0.01, balancing_x_dim=5, layer_dims_balancing_x=[], layer_dims_output=[200,200,200], dropout_prob=0., weight_decay=0., optimizer="adam", activation="relu", final_activation_output='exp', **kwargs):
        super().__init__()

        self.final_activation_output = final_activation_output
        final_activation_output_bool = final_activation_output if isinstance(final_activation_output, bool) else False

        final_layer_x = [] if balancing_x_dim is None else [balancing_x_dim]
        self.score_x = build_fc_network([x_dim] + layer_dims_balancing_x + final_layer_x, dropout_prob=dropout_prob, activation=activation, final_activation=False)
        self.intermediary_dim = ([x_dim] + layer_dims_balancing_x + final_layer_x)[-1]

        self.output_net = nn.ModuleList([
            build_fc_network([self.intermediary_dim] + layer_dims_output + [1], dropout_prob=dropout_prob, activation=activation, final_activation=final_activation_output_bool) for _ in range(a_dim)
        ])

        self.activation = activation
        self.optimizer = optimizer
        self.init_lr = init_lr
        self.weight_decay = weight_decay

        for kwarg in kwargs:
            print(f'WARNING : {kwarg} not used by BalancingScoreNet.')

    def training_step(self, batch, batch_idx, name='train_loss'):
        x, a, groundtruth = batch
        x_rep = ACTIVATIONS[self.activation]()(self.score_x(x))
        prediction = torch.sum(torch.cat([
            torch.reshape(output(x_rep), (-1,1)) for output in self.output_net
        ], dim=-1) * a, dim=-1)
        if isinstance(self.final_activation_output, str) and self.final_activation_output == 'exp':
            prediction = torch.exp(prediction)
        groundtruth = torch.flatten(groundtruth)
        loss = nn.functional.mse_loss(groundtruth, prediction)
        self.log(name, loss)
        return loss

    def validation_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx, name='val_loss')

    def configure_optimizers(self):
        # define optimizer and scheduler
        optimizer = OPTIMIZERS[self.optimizer](
            filter(lambda p: p.requires_grad, self.parameters()), lr=self.init_lr,weight_decay=self.weight_decay
        )

        # training and evaluation loop
        epoch_lr = optimizer.param_groups[0]['lr']

        # TODO add support for learning rate scheduler
        # adjust learning rate
        # if epoch % args.update_lr_every_epoch == 0 and not epoch == 0:
        #     adjust_learning_rate(optimizer, epoch_lr, args.lr_decay, args.min_lr)

        return optimizer


class BalancingScoreNetDifferentScores(pl.LightningModule):
    def __init__(self, x_dim, a_dim, init_lr=0.01, balancing_x_dim=5, layer_dims_balancing_x=[], layer_dims_output=[200,200,200], dropout_prob=0., weight_decay=0., optimizer="adam", activation="relu", final_activation_output='exp', **kwargs):
        super().__init__()

        self.final_activation_output = final_activation_output
        final_activation_output_bool = final_activation_output if isinstance(final_activation_output, bool) else False

        final_layer_x = [] if balancing_x_dim is None else [balancing_x_dim]
        self.scores_x = nn.ModuleList([
            build_fc_network([x_dim] + layer_dims_balancing_x + final_layer_x, dropout_prob=dropout_prob, activation=activation, final_activation=False) for _ in range(a_dim)
        ])
        self.intermediary_dim = ([x_dim] + layer_dims_balancing_x + final_layer_x)[-1]

        self.output_net = nn.ModuleList([
            build_fc_network([self.intermediary_dim] + layer_dims_output + [1], dropout_prob=dropout_prob, activation=activation, final_activation=final_activation_output_bool) for _ in range(a_dim)
        ])

        self.activation = activation
        self.optimizer = optimizer
        self.init_lr = init_lr
        self.weight_decay = weight_decay

        for kwarg in kwargs:
            print(f'WARNING : {kwarg} not used by BalancingScoreNet.')

    def score_x(self, x, a):
        reprs = torch.cat([
            torch.reshape(score(x), (x.shape[0],  self.intermediary_dim, 1)) for score in self.scores_x
        ], dim=-1)
        a_duplicated = a.unsqueeze(1).repeat(1, self.intermediary_dim, 1)
        repr = torch.reshape(torch.sum(reprs * a_duplicated, axis=-1), (x.shape[0],  self.intermediary_dim))
        return repr

    def training_step(self, batch, batch_idx, name='train_loss'):
        x, a, groundtruth = batch
        x_rep = ACTIVATIONS[self.activation]()(self.score_x(x, a))
        prediction = torch.sum(torch.cat([
            torch.reshape(output(x_rep), (-1,1)) for output in self.output_net
        ], dim=-1) * a, dim=-1)
        if isinstance(self.final_activation_output, str) and self.final_activation_output == 'exp':
            prediction = torch.exp(prediction)
        groundtruth = torch.flatten(groundtruth)
        loss = nn.functional.mse_loss(groundtruth, prediction)
        self.log(name, loss)
        return loss

    def validation_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx, name='val_loss')

    def configure_optimizers(self):
        # define optimizer and scheduler
        optimizer = OPTIMIZERS[self.optimizer](
            filter(lambda p: p.requires_grad, self.parameters()), lr=self.init_lr,weight_decay=self.weight_decay
        )

        # training and evaluation loop
        epoch_lr = optimizer.param_groups[0]['lr']

        # TODO add support for learning rate scheduler
        # adjust learning rate
        # if epoch % args.update_lr_every_epoch == 0 and not epoch == 0:
        #     adjust_learning_rate(optimizer, epoch_lr, args.lr_decay, args.min_lr)

        return optimizer

