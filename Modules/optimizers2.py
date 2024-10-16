import torch
import torch.optim as optim
import torch.nn as nn


class VectorOptim(nn.Module):
    """ Class to simplify the vector optimization using a given optimizer. """

    def __init__(self, network_parameters, lr=2e-4, weight_decay=0, optimizer_name="Adam", shuffled=False,
                 cuda_scaler=False, *args, **kwargs):
        """
        :param network_parameters: The parameters of the network to optimizer. use net.parameters()
        :param learning_rate: The learning rate. Default is 2e-4
        :param weight_decay:
        :param optimizer_name: name of the optimizer to use. By default we use Adam
        :param shuffled: shuffle the vector before calling backward. """

        super().__init__()
        self.shuffled = shuffled
        self.optimizer = getattr(optim, optimizer_name)
        self.optimizer = self.optimizer(params=network_parameters, lr=lr, weight_decay=weight_decay,
                                        *args, **kwargs)
        self.step = self.step_15
        if "1.4" in torch.__version__:
            self.step = self.step_14
        self.scaler = torch.cuda.amp.GradScaler(enabled=cuda_scaler)

    def step_14(self, losses):
        """
        Update the network parameters. Do the gradient descent
        :param losses: List of losses to update the model with
        """
        for loss in losses:
            loss = loss.mean(0)
            loss = loss.view(-1)
            if self.shuffled:
                loss = loss[torch.randperm(loss.size(0))]
            for l in loss.view(-1, 1):
                self.optimizer.zero_grad()
                l.backward(retain_graph=True)
                self.optimizer.step()

    def step_15(self, losses):
        """
        Update the network parameters. Do the gradient descent
        :param losses: List of losses to update the model with
        """
        self.backward(losses)
        self.optim_step()

    def backward(self, losses, *args, **kwargs):
        """
        Just compute the backward pass
        """
        self.optimizer.zero_grad()
        for loss in losses:
            loss = loss.mean(0)
            loss = loss.view(-1)
            if self.shuffled:
                loss = loss[torch.randperm(loss.size(0))]
            for l in loss.view(-1, 1):
                self.scaler.scale(l).backward(retain_graph=True, *args, **kwargs)

    def zero_grad(self):
        self.optimizer.zero_grad(set_to_none=True)

    def optim_step(self):
        """ Step pass """
        self.scaler.step(self.optimizer)
        self.scaler.update()

    def __str__(self):
        return "{}(\n {}\nShuffled: {})".format(self.__class__.__name__, str(self.optimizer), self.shuffled)


class DualVectorOptim(nn.Module):
    """ Class to simplify the vector optimization using a given optimizer. """

    def __init__(self, network_parameters, lr=2e-4, weight_decay=0, optimizer_name="AdamW", shuffled=None,
                 cuda_scaler=False, *args, **kwargs):
        """
        :param network_parameters: The parameters of the network to optimizer. use net.parameters()
        :param learning_rate: The learning rate. Default is 2e-4
        :param weight_decay:
        :param optimizer_name: name of the optimizer to use. By default we use Adam
        :param shuffled: shuffle the vector before calling backward."""
        super().__init__()
        self.optimizer = getattr(optim, optimizer_name)
        self.optimizer = self.optimizer(params=network_parameters, lr=lr, weight_decay=weight_decay,
                                        *args, **kwargs)
        self.step = self.step_15
        if "1.4" in torch.__version__:
            self.step = self.step_14
        self.scaler = torch.cuda.amp.GradScaler(enabled=cuda_scaler)

    def step_14(self, losses):
        """
        Update the network parameters. Do the gradient descent
        :param losses: List of losses to update the model with
        """
        for loss in losses:
            loss = loss.mean()
            for l in loss.view(-1, 1):
                self.optimizer.zero_grad()
                l.backward(retain_graph=True)
                self.optimizer.step()

    def step_15(self, losses):
        """
        Update the network parameters. Do the gradient descent
        :param losses: List of losses to update the model with
        """
        self.backward(losses)
        self.optim_step()

    def backward(self, losses, *args, **kwargs):
        """
        Just compute the backward pass
        """
        # self.optimizer.zero_grad()
        for loss in losses:
            loss = loss.mean()
            for l in loss.view(-1, 1):
                # Est ce que backward va poser pb si c'est dict ?
                # Checker les param`etres pour voir si backward avec input donne la mme chose que sans ?
                # Mais comment verifier les milliers de params ?
                self.scaler.scale(l).backward(retain_graph=True, *args, **kwargs)

    def zero_grad(self):
        self.optimizer.zero_grad(set_to_none=True)

    def optim_step(self):
        """ Step pass """
        self.scaler.step(self.optimizer)
        self.scaler.update()

    def __str__(self):
        return "{}(\n {})".format(self.__class__.__name__, str(self.optimizer))


class OptimizerWrapper(nn.Module):
    """ Wrapper on the optimizer used with torch. """

    def __init__(self, network_parameters, lr=2e-4, weight_decay=0, optimizer_name="AdamW", shuffled=None,
                 retain_graph=False, cuda_scaler=False, *args, **kwargs):
        """
        :param network_parameters: The parameters of the network to optimizer. use net.parameters()
        :param learning_rate: The learning rate. Default is 2e-4
        :param weight_decay:
        :param optimizer_name: name of the optimizer to use. By default we use Adam
        :param retain_graph: keep the computationnal graph alive of remove it at the first backward call
        """
        super().__init__()
        self.optimizer = getattr(optim, optimizer_name)
        self.retain_graph = retain_graph
        self.optimizer = self.optimizer(params=network_parameters, lr=lr, weight_decay=weight_decay,
                                        *args, **kwargs)
        self.scaler = None
        self.scaler = torch.cuda.amp.GradScaler(enabled=cuda_scaler)

    def step(self, losses):
        """
        Update the network parameters. Do the gradient descent
        :param loss: computed loss
        """
        self.backward(losses)
        self.optim_step()

    def backward(self, losses, *args, **kwargs):
        """
        Just compute the backward pass
        """
        if isinstance(losses, list):
            losses = losses[0]
        # self.optimizer.zero_grad()
        self.scaler.scale(losses).backward(retain_graph=self.retain_graph, *args, **kwargs)

    def zero_grad(self):
        self.optimizer.zero_grad()

    def optim_step(self):
        """ Step pass """
        self.scaler.step(self.optimizer)
        self.scaler.update()

    def __str__(self):
        return "{}(\n {}\nRetain Graph: {})".format(self.__class__.__name__, str(self.optimizer), self.retain_graph)
