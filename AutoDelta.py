from numpyro.contrib.autoguide import AutoGuide
from numpyro.infer import init_to_median
from numpyro import handlers
from abc import ABC, abstractmethod
import warnings
import operator
from jax import hessian, random, vmap
from jax.experimental import stax
from jax.flatten_util import ravel_pytree
import jax.numpy as np
from jax.tree_util import tree_map
from contextlib import ExitStack  # python 3

import numpyro
from numpyro.contrib.nn.auto_reg_nn import AutoregressiveNN
from numpyro.contrib.nn.block_neural_arn import BlockNeuralAutoregressiveNN
import numpyro.distributions as dist
from numpyro.distributions import constraints
from numpyro.distributions.flows import BlockNeuralAutoregressiveTransform, InverseAutoregressiveTransform
from numpyro.distributions.transforms import (
    AffineTransform,
    ComposeTransform,
    MultivariateAffineTransform,
    PermuteTransform,
    UnpackTransform,
    biject_to
)
from numpyro.distributions.util import cholesky_of_inverse, sum_rightmost
from numpyro.handlers import seed, substitute
from numpyro.infer.elbo import ELBO
from numpyro.infer.util import constrain_fn, find_valid_initial_params, init_to_uniform, log_density, transform_fn
from numpyro.util import not_jax_tracer

class AutoDelta(AutoGuide):
    """
    This implementation of :class:`AutoGuide` uses Delta distributions to
    construct a MAP guide over the entire latent space. The guide does not
    depend on the model's ``*args, **kwargs``.
    .. note:: This class does MAP inference in constrained space.
    Usage::
        guide = AutoDelta(model)
        svi = SVI(model, guide, ...)
    Latent variables are initialized using ``init_loc_fn()``. To change the
    default behavior, create a custom ``init_loc_fn()`` as described in
    :ref:`autoguide-initialization` , for example::
        def my_init_fn(site):
            if site["name"] == "level":
                return torch.tensor([-1., 0., 1.])
            if site["name"] == "concentration":
                return torch.ones(k)
            return init_to_sample(site)
    :param callable model: A Pyro model.
    :param callable init_loc_fn: A per-site initialization function.
        See :ref:`autoguide-initialization` section for available functions.
    """
    def __init__(self, model, init_loc_fn=init_to_median):
        self.init_loc_fn = init_loc_fn
        #model = InitMessenger(self.init_loc_fn)(model)
        super().__init__(model)

    def _setup_prototype(self, *args, **kwargs):
        super()._setup_prototype(*args, **kwargs)
        rng_key = numpyro.sample("_{}_rng_key_init".format(self.prefix), dist.PRNGIdentity())
        init_params, _ = handlers.block(find_valid_initial_params)(rng_key, self.model,
                                                                   init_strategy=self.init_strategy,
                                                                   model_args=args,
                                                                   model_kwargs=kwargs)
        print(init_params)
        # Initialize guide params
        # for name, site in self.prototype_trace.iter_stochastic_nodes():
        #     value = PyroParam(site["value"].detach(), constraint=site["fn"].support)_deep_setattr(self, name, value)

    def __call__(self, *args, **kwargs):
        """
        An automatic guide with the same ``*args, **kwargs`` as the base ``model``.
        :return: A dict mapping sample site name to sampled value.
        :rtype: dict
        """
        # if we've never run the model before, do so now so we can inspect the model structure
        if self.prototype_trace is None:
            self._setup_prototype(*args, **kwargs)

        plates = self._create_plates()
        result = {}
        for name, site in self.prototype_trace.iter_stochastic_nodes():
            with ExitStack() as stack:
                for frame in site["cond_indep_stack"]:
                    if frame.vectorized:
                        stack.enter_context(plates[frame.name])
                attr_get = operator.attrgetter(name)
                result[name] = numpyro.sample(name, dist.Delta(attr_get(self),
                                                            event_dim=site["fn"].event_dim))
        return result

    def median(self, *args, **kwargs):
        """
        Returns the posterior median value of each latent variable.
        :return: A dict mapping sample site name to median tensor.
        :rtype: dict
        """
        return self(*args, **kwargs)

    def _sample_latent(self, *args, **kwargs):
        pass
    def sample_posterior(self, rng_key, params, *args, **kwargs):
        pass

