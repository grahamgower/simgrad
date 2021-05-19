from typing import NamedTuple, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import tskit


class SizeChangeEvent(NamedTuple):
    t: float
    x: float


def simulate(events, n, key: Tuple[int]):
    """
    Simulate the Kingman coalescent for a sample of size :math:`n`, returning
    an oriented forest :math:`\\pi` and node-time array :math:`\\tau`.
    This is based on the algorithm from Hudson 1990.

    :param list events: A list of size-change events. Each event is a 2-tuple
        of :math:`(t, x)`, where :math:`t` is the time at which the population
        size is set to :math:`x N_0`.
    :param int n: The number of haploid samples.
    :param tuple(int, int) key: JAX random number state.
    """
    events = events + [SizeChangeEvent(np.inf, 1.0)]
    event_index = 0
    pi = [-1] * (2 * n - 1)
    tau = [0] * (2 * n - 1)
    lineages = list(range(n))
    j = n - 1
    t = 0.0
    x = 1.0
    parent = n

    while j > 0:
        rate = (j + 1) * j / 2.0
        key, subkey = jax.random.split(key)
        t += x * jax.random.exponential(subkey) / rate
        if events[event_index].t < t:
            # size-change event
            t, x = events[event_index]
            event_index += 1
        else:
            # coalescent event
            tau[parent] = t
            key, subkey1, subkey2 = jax.random.split(key, 3)
            # child 1
            k = jax.random.randint(subkey1, shape=[1], minval=0, maxval=j)[0]
            pi[lineages[k]] = parent
            lineages[k] = lineages[j]
            j -= 1
            # child 2
            k = jax.random.randint(subkey2, shape=[1], minval=0, maxval=j)[0]
            pi[lineages[k]] = parent
            lineages[k] = parent
            parent += 1

    return jnp.array(pi), jnp.array(tau)


def branch_sfs(pi, tau):
    """
    Branch-length analogue of the SFS.
    """
    assert len(pi) == len(tau)
    n = (len(tau) + 1) // 2
    # We include "0" and "n" bins to match tskit allele_frequency_spectrum().
    sfs = [0] * (n + 1)
    num_below = [1] * n + [0] * (n - 1)
    for child, parent in enumerate(pi[:-1]):
        num_below[parent] += num_below[child]
        sfs[num_below[child]] += tau[parent] - tau[child]
    return jnp.array(sfs)


def loss(*, target_sfs, events, n, key, nreps=20):
    """
    Mean squared error between ``target_sfs`` and simulated sfs
    when simulating with the given ``events``, averaged over ``nreps``
    replicate simulations.
    """
    avg_sfs = jnp.zeros(n + 1)
    for key in jax.random.split(key, nreps):
        pi, tau = simulate(events, n, key)
        sfs = branch_sfs(pi, tau)
        avg_sfs = jnp.add(avg_sfs, sfs)
    avg_sfs /= nreps
    sumsq = jnp.sum(jnp.subtract(avg_sfs, target_sfs) ** 2)
    return sumsq


def to_tskit(pi, tau):
    assert len(pi) == len(tau)
    n = (len(tau) + 1) // 2
    tables = tskit.TableCollection(1)
    for j, time in enumerate(tau):
        flags = tskit.NODE_IS_SAMPLE if j < n else 0
        tables.nodes.add_row(flags, time)
    for child, parent in enumerate(pi):
        if parent != tskit.NULL:
            tables.edges.add_row(0, 1, parent, child)
    tables.sort()
    return tables.tree_sequence()


def create_target_sfs(events, n, key, nreps=100):
    """
    Simulate the sfs for a test scenario, as a target for evaluating
    the inference process.
    """
    avg_sfs = np.zeros(n + 1)
    for _ in range(nreps):
        key, subkey = jax.random.split(key)
        pi, tau = simulate(events, n, subkey)
        # print(ts.draw_text())
        sfs = branch_sfs(pi, tau)
        avg_sfs += np.array(sfs)

        # check that our branch_sfs() function matches tskit
        ts = to_tskit(pi, tau)
        ts_sfs = ts.allele_frequency_spectrum(mode="branch", polarised=True)
        np.testing.assert_allclose(sfs, ts_sfs, rtol=1e-4)

    avg_sfs /= nreps
    return avg_sfs


if __name__ == "__main__":
    sample_size = 10
    key = jax.random.PRNGKey(314159)

    # Parameters of test scenario to evaluate the inference process.
    x_true = 10.0
    t_true = 0.05
    print("x_true, t_true:", (x_true, t_true))

    key, subkey = jax.random.split(key)
    target_sfs = create_target_sfs(
        [SizeChangeEvent(t_true, x_true)], sample_size, subkey
    )
    print("target_sfs:", target_sfs)

    # Look at a grid of values of ``x``.
    myloss = lambda x, key: loss(
        events=[SizeChangeEvent(t_true, x)],
        target_sfs=target_sfs,
        key=key,
        n=sample_size,
    )
    for x in np.exp(np.linspace(np.log(1), np.log(100), num=20)):
        key, subkey = jax.random.split(key)
        value, grad = jax.value_and_grad(myloss)(x, subkey)
        print(x, value, grad)
