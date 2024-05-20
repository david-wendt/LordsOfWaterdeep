# From assignment 2

import torch
import torch.nn as nn
import torch.distributions as ptd

from network_utils import np2torch, device


class BasePolicy:
    def action_distribution(self, observations):
        """
        Args:
            observations: torch.Tensor of shape [batch size, dim(observation space)]
        Returns:
            distribution: instance of a subclass of torch.distributions.Distribution

        See https://pytorch.org/docs/stable/distributions.html#distribution

        This is an abstract method and must be overridden by subclasses.
        It will return an object representing the policy's conditional
        distribution(s) given the observations. The distribution will have a
        batch shape matching that of observations, to allow for a different
        distribution for each observation in the batch.
        """
        raise NotImplementedError

    def act(self, observations, return_log_prob = False):
        """
        Args:
            observations: np.array of shape [batch size, dim(observation space)]
        Returns:
            sampled_actions: np.array of shape [batch size, *shape of action]
            log_probs: np.array of shape [batch size] (optionally, if return_log_prob)

        TODO:
        Call self.action_distribution to get the distribution over actions,
        then sample from that distribution. Compute the log probability of
        the sampled actions using self.action_distribution. You will have to
        convert the actions and log probabilities to a numpy array, via numpy(). 

        You may find the following documentation helpful:
        https://pytorch.org/docs/stable/distributions.html
        """
        observations = np2torch(observations)
        #######################################################
        #########   YOUR CODE HERE - 1-4 lines.    ############
        distribution = self.action_distribution(observations)
        sampled_actions = distribution.sample()

        log_probs = distribution.log_prob(sampled_actions)

        sampled_actions = sampled_actions.numpy()
        log_probs = log_probs.detach().numpy()


        # Call self.action_distribution to get the distribution over actions,
        # then sample from that distribution
        # sampled_actions = self.action_distribution(observations).sample()
        # Compute the log probability of
        # the sampled actions using self.action_distribution
        # distribution = self.action_distribution.log_prob(sampled_actions)

        # look back at this .detatch.cpu()?
        # sampled_actions = sampled_actions.numpy()
        # log_probs = log_probs.numpy()

        #######################################################
        #########          END YOUR CODE.          ############
        if return_log_prob:
            return sampled_actions, log_probs
        return sampled_actions


class CategoricalPolicy(BasePolicy, nn.Module):
    def __init__(self, network):
        nn.Module.__init__(self)
        self.network = network

    def action_distribution(self, observations):
        """
        Args:
            observations: torch.Tensor of shape [batch size, dim(observation space)]
        Returns:
            distribution: torch.distributions.Categorical where the logits
                are computed by self.network

        See https://pytorch.org/docs/stable/distributions.html#categorical
        """
        #######################################################
        #########   YOUR CODE HERE - 1-2 lines.    ############
        logits = self.network(observations)
        probabilities = torch.softmax(logits, dim=-1)
        distribution = torch.distributions.Categorical(probabilities)
        #######################################################
        #########          END YOUR CODE.          ############
        return distribution