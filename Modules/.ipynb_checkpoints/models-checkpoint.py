import os
from abc import ABC

import torch
import torch.nn as nn


class GenericNN(nn.Module):

    def __init__(self):

        super().__init__()
        self.loaded = False
        self.extension = ""
        self.optimizer = None

    def reset_weights(self):
        """ Apply the function to reset weights """
        self.apply(self.__reset_model_weights__)

    @staticmethod
    def __reset_model_weights__(m):
        """ Reset the models parameters. """
        try:
            m.reset_parameters()
        except AttributeError as e:
            pass

    def save_model_state(self, version, directory, param_fn):
        """
        Save given classifier parameter
        :param self: Classifier to save
        :param version: classifier parameter version (epoch number)
        :param directory: destination directory, where to save parameters.
        :param param_fn: Function to add other parameters on the name of the model
        :return: False if there is an error. True otherwise
        """
        if param_fn is None:
            param_fn = lambda x: x
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(self.state_dict(), "{d}/Epoch_{v}.{e}".format(d=directory, v=param_fn(version), e=self.extension))

    def load_model_state(self, epoch, model_dir, param_fn):
        """
        Load classifier parameter
        :param self: Classifier where to load parameters.
        :param epoch: epoch to load
        :param model_dir: directory where to load epochs. First directory
        """
        if param_fn is None:
            param_fn = lambda x: x
        self.load_state_dict(torch.load("{d}/Epoch_{v}.{e}".format(d=model_dir, v=param_fn(epoch), e=self.extension)))
        self.loaded = True

    def check_state(self, epoch, model_dir, param_fn):
        """
        Check if the given epoch exists for the model
        """
        good = False
        if os.path.isfile("{d}/Epoch_{v}.{e}".format(d=model_dir, v=param_fn(epoch), e=self.extension)):
            good = check_parameters(self, epoch, model_dir, param_fn)
        return good

    def delete_model_state(self, epoch, model_dir, param_fn):
        """
        Load classifier parameter
        :param self: Classifier where to load parameters.
        :param epoch: epoch to load
        :param model_dir: directory where to load epochs. First directory
        """
        if param_fn is None:
            param_fn = lambda x: x
        try:
            os.remove("{d}/Epoch_{v}.{e}".format(d=model_dir, v=param_fn(epoch), e=self.extension))
        except FileNotFoundError:
            pass


class Predictor(GenericNN):

    def __init__(self, input_dim, predictor_dim, output_dim=1, pack=10, output_probs="log_softmax"):
        super(Predictor, self).__init__()
        dim = input_dim * pack
        self.pack = pack
        self.packdim = dim
        self.output_dim = output_dim
        seq = []
        for item in list(predictor_dim):
            seq += [nn.Linear(dim, item), nn.LeakyReLU(0.2), nn.Dropout(0.5)]
            dim = item

        seq += [nn.Linear(dim, output_dim)]
        if output_probs == "log_softmax":
            seq.append(nn.LogSoftmax(dim=1))
        elif output_probs == "sigmoid":
            seq.append(nn.Sigmoid())
        elif output_probs == "softmax":
            seq.append(nn.Softmax(dim=1))
        self.seq = nn.Sequential(*seq)

    def forward(self, input):
        assert input.size()[0] % self.pack == 0
        return self.seq(input.view(-1, self.packdim))


class CTGANDiscriminator(GenericNN):

    def __init__(self, input_dim, discriminator_dim, output_dim=1, pack=10, softmax=False):
        super(CTGANDiscriminator, self).__init__()
        dim = input_dim * pack
        self.pack = pack
        self.packdim = dim
        self.output_dim = output_dim
        seq = []
        for item in list(discriminator_dim):
            seq += [nn.Linear(dim, item), nn.LeakyReLU(0.2), nn.Dropout(0.5)]
            dim = item

        seq += [nn.Linear(dim, output_dim)]
        if softmax:
            seq += [nn.Softmax(dim=1)]
        self.seq = nn.Sequential(*seq)
        self.penalty = self.calc_gradient_penalty

    def calc_gradient_penalty(self, real_data, fake_data, device='cpu', pac=10, lambda_=10, **irrelevant):
        alpha = torch.rand(real_data.size(0) // pac, 1, 1, device=device)
        alpha = alpha.repeat(1, pac, real_data.size(1))
        alpha = alpha.view(-1, real_data.size(1))

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        disc_interpolates = self(interpolates)

        gradients = torch.autograd.grad(
            outputs=disc_interpolates, inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size(), device=device),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]

        gradient_penalty = ((
                                    gradients.view(-1, pac * real_data.size(1)).norm(2, dim=1) - 1
                            ) ** 2).mean() * lambda_

        return gradient_penalty

    def calc_divergence(self, real_data, fake_data, real_validity, fake_validity, device="cpu", pac=10, k=2, p=6,
                        **irrelevant):
        # Compute W-div gradient penalty
        real_grad_out = torch.ones(real_data.size(0), 1, device=device, requires_grad=False)
        # real_data = real_data.view(-1, self.packdim)
        # real_grad_out = torch.ones(real_data.size(0) // pac, 1, 1, device=device, requires_grad=False)
        # real_grad_out = real_grad_out.repeat(1, pac, real_data.size(1))
        # real_grad_out = real_grad_out.view(-1, real_data.size(1))
        fake_grad_out = torch.ones(fake_data.size(0), 1, device=device, requires_grad=False)
        # fake_data = fake_data.view(-1, self.packdim)
        # fake_grad_out = torch.ones(fake_data.size(0) // pac, 1, 1, device=device, requires_grad=False)
        # fake_grad_out = fake_grad_out.repeat(1, pac, fake_data.size(1))
        # fake_grad_out = fake_grad_out.view(-1, fake_data.size(1))

        real_grad = torch.autograd.grad(
            real_validity, real_data, real_grad_out, create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        real_grad_norm = real_grad.view(real_grad.size(0), -1).pow(2).sum(1) ** (p / 2)

        fake_grad = torch.autograd.grad(
            fake_validity, fake_data, fake_grad_out, create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        fake_grad_norm = fake_grad.view(fake_grad.size(0), -1).pow(2).sum(1) ** (p / 2)

        div_gp = torch.mean(real_grad_norm + fake_grad_norm) * k / 2
        return div_gp

    def forward(self, input):
        assert input.size()[0] % self.pack == 0
        return self.seq(input.view(-1, self.packdim))

    
class FMDiscriminator(GenericNN):

    def __init__(self, input_dim, discriminator_dim, output_dim=1, pack=10, softmax=False):
        super(FMDiscriminator, self).__init__()
        dim = input_dim * pack
        self.pack = pack
        self.packdim = dim
        self.output_dim = output_dim
        common = []
        for item in list(discriminator_dim):
            common += [nn.Linear(dim, item), nn.LeakyReLU(0.2), nn.Dropout(0.5)]
            dim = item

        # common += [nn.Linear(dim, output_dim)]
        groups = [nn.Linear(dim, output_dim)]
        critic = [nn.Linear(dim, 1)]
        if softmax:
            groups += [nn.Softmax(dim=1)]
        self.common = nn.Sequential(*common)
        self.groups = nn.Sequential(*groups)
        self.critic = nn.Sequential(*critic)
        

    def forward(self, input):
        assert input.size()[0] % self.pack == 0
        common_out = self.common(input.view(-1, self.packdim))
        critic_out = self.critic(common_out)
        groups_out = self.groups(common_out)
        return critic_out, groups_out
    

class CTGANResidual(nn.Module):

    def __init__(self, i, o, bn=True):
        super(CTGANResidual, self).__init__()
        self.seq = [nn.Linear(i, o)]
        if bn:
            self.seq.append(nn.BatchNorm1d(o))
            # self.bn = nn.BatchNorm1d(o)
        self.seq.append(nn.ReLU())
        # self.relu = nn.ReLU()
        self.seq = nn.Sequential(*self.seq)

    def forward(self, input):
        # out = self.fc(input)
        # out = self.bn(out)
        # out = self.relu(out)
        out = self.seq(input)
        return torch.cat([out, input], dim=1)


class CTGANGenerator(GenericNN):

    def __init__(self, embedding_dim, generator_dim, data_dim, tanh_output=False, bn=True):
        super(CTGANGenerator, self).__init__()
        dim = embedding_dim
        seq = []
        for item in list(generator_dim):
            seq += [CTGANResidual(dim, item, bn=bn)]
            dim += item
        seq.append(nn.Linear(dim, data_dim))
        if tanh_output:
            seq.append(nn.Tanh())
        self.seq = nn.Sequential(*seq)

    def forward(self, input, controls=None):
        if controls is not None:
            input = torch.cat((input, controls), 1)
        data = self.seq(input)
        return data


# Discriminator
class Discriminator(GenericNN):
    """
    Discriminator class
    """

    def __init__(self, input_size, dsc_output_size=1, clf_output_size=1):
        """
        """
        super().__init__()
        common_output_size = input_size
        common_modules = [
            nn.Linear(input_size, input_size * 4),
            nn.LeakyReLU(),
            nn.Linear(input_size * 4, input_size * 2),
            nn.LeakyReLU(),
            nn.Linear(input_size * 2, common_output_size),
        ]
        dsc_modules = [
            nn.Linear(common_output_size, dsc_output_size),
            nn.LeakyReLU()
        ]
        clf_modules = [
            nn.Linear(common_output_size, clf_output_size),
            nn.LeakyReLU()
        ]
        self.common = nn.Sequential(*common_modules)
        self.dsc_out = nn.Sequential(*dsc_modules)
        self.clf_out = nn.Sequential(*clf_modules)

        self.sigmoid_out = nn.Sigmoid()

    def forward(self, input_):
        common = self.common(input_)
        return self.dsc_out(common), self.clf_out(common)

    def predict(self, x):
        dsc, clf = self.forward(x)
        return self.sigmoid_out(dsc), self.sigmoid_out(clf)


# Wrapper to transform the single module output into several independent parts
class DiscriminatorSingleToMultiOutput(Discriminator):

    def __init__(self, targets_len, input_size, output_size):
        """
        Init the class
        :param target_len: list of respective target nodes length. For instance, if we use 2 attributes, we should
         expect targets_len = [2, 3] in example. Meaning that the first attribute will use the first two nodes, while
         the second attribute will use the last 3 nodes.
        """
        super().__init__(input_size, output_size)
        self.targets_len = targets_len
        self.priority_output = None  # Select a specific output and disregard the rest.

    def forward(self, *args, **kwargs):
        output = super(DiscriminatorSingleToMultiOutput, self).forward(*args, **kwargs)
        outputs = []
        start = 0
        for i in self.targets_len:
            outputs.append(output[:, start: start + i])
            start += i
        return outputs

    def predict(self, *args, **kwargs):
        outputs = self.forward(*args, **kwargs)
        if self.priority_output is not None:
            return outputs[self.priority_output]
        return outputs


# Generator Enc
class GeneratorEnc(GenericNN):
    """
    Sanitizer class.
    """

    def __init__(self, input_size, output_size):
        """
        """

        super().__init__()
        modules = [
            nn.Linear(input_size, input_size),
            nn.ReLU(),
            nn.Linear(input_size, input_size),
            nn.LeakyReLU(),
            nn.Linear(input_size, output_size),
        ]
        self.seq_linears = nn.Sequential(*modules)

    def forward(self, input_):
        input_ = self.seq_linears(input_)
        return input_


# Generator Dec
class GeneratorDec(GenericNN):
    """
    Sanitizer class.
    """

    def __init__(self, input_size, output_size):
        """
        """

        super().__init__()
        modules = [
            nn.Linear(input_size, input_size),
            nn.ReLU(),
            nn.Linear(input_size, input_size),
            nn.LeakyReLU(),
            nn.Linear(input_size, output_size),
        ]
        self.seq_linears = nn.Sequential(*modules)

    def forward(self, input_, control):
        input_ = torch.cat((input_, control), 1)
        input_ = self.seq_linears(input_)
        return input_


class EncoderDecoder(GenericNN):
    """ Encoder Model """


class EncoderDecoder(GenericNN):
    """ Encoder Model """

    def __init__(self, input_size, compression_size):
        """ """
        super().__init__()
        modules = [
            nn.Linear(input_size, input_size),
            nn.ReLU(),
            nn.Linear(input_size, compression_size),
            nn.LeakyReLU(),
        ]
        self.seq_linears = nn.Sequential(*modules)

    def forward(self, input_):
        input_ = self.seq_linears(input_)
        return input_


def check_parameters(model, *load_model_args, **load_model_kwargs):
    """
    Model parameters must be different from nan.
    :return: True if params are all different from nan
    """
    try:
        model.load_model_state(*load_model_args, **load_model_kwargs)
    except (EOFError, RuntimeError) as e:
        # The pickle file containing the model information is empty or corrupted.
        return False
    for p in model.parameters():
        if torch.isnan(p).any().data.item():
            return False
    return True


def get_latest_consistent_epoch(model_dir, models, epoch_step=1, param_fn=None):
    """
    Return the latest epoch where both the sanitizer and all predictors are available
    """
    epoch = 0
    stop = False
    c_format = "{d}/Epoch_{v}.{e}"
    while not stop:
        epoch += epoch_step
        for model in models:
            stop = not os.path.isfile(c_format.format(d=model_dir, v=param_fn(epoch), e=model.extension)) or stop
            if not stop:
                # The file does exists
                stop = not check_parameters(model, epoch, model_dir, param_fn)
            if stop:
                break
    return epoch - epoch_step


def get_latest_states(model_dir, models, param_fn=None, epoch_step=1):
    """
    Get the latest saved states of models
    :param model_dir: location of models
    :param models: list of models to check for
    :param extensions: extensions of the given models, in the same order as models
    :param param_fn: add params values in string
    :param epoch_step: step to check for valid elements. Usefull if we have a large number of epoch (e.g: 200)
    :return: The closest epoch at which no models exists.
    """
    epoch = get_latest_consistent_epoch(model_dir=model_dir, models=models, epoch_step=epoch_step, param_fn=param_fn)
    if epoch > 0:
        for model in models:
            model.load_model_state(epoch, model_dir, param_fn)
    return epoch + 1


def dump_models_parameters(models_list, optimizers_list, noise_nodes, noise_generator, criterions_list,
                           batch_size_list, iteration_list, reset_models, dim_reduction, alpha_list,
                           k_folds, path, filename="experiments_parameters.txt"):
    """ Dump all of the experiments parameters. """
    with open("{}/{}".format(path, filename), "w") as dmp:
        for m, o, c, i, b in zip(models_list, optimizers_list, criterions_list, iteration_list, batch_size_list):
            dmp.write(str(m))
            dmp.write("\nOptimizer\n")
            dmp.write(str(o))
            dmp.write("\nLoss Fn\n")
            dmp.write(str(c))
            dmp.write("\nModels Iterations: {}".format(i))
            dmp.write("\nBatch Size for Model: {}\n".format(b))
        dmp.write("Nb Noise Nodes: {}\n".format(noise_nodes))
        dmp.write("Noise generator: {}\n".format(noise_generator))
        dmp.write("Reset (Disc) Models: {}\n".format(reset_models))
        dmp.write("Dimensionality Reduction: {}\n".format(dim_reduction))
        dmp.write("Alpha: {}\n".format(alpha_list))
        dmp.write("Nb Folds: {}\n".format(k_folds))
