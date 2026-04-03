Quick Start
===========

Drop-in SAMC with SAMCWeights
------------------------------

Add two lines to any Metropolis-Hastings loop:

.. code-block:: python

    from illuma_samc import SAMCWeights

    wm = SAMCWeights()

    for t in range(1, n_steps + 1):
        x_new = propose(x)
        fy = energy_fn(x_new)

        log_r = (-fy + fx) / T + wm.correction(fx, fy)
        if log_r > 0 or math.log(random()) < log_r:
            x, fx = x_new, fy
        wm.step(t, fx)

Batteries-included SAMC
------------------------

.. code-block:: python

    from illuma_samc import SAMC

    sampler = SAMC(energy_fn=my_energy, dim=2)
    result = sampler.run(n_steps=100_000)

Using SAMCConfig
-----------------

Reduce boilerplate with a config object:

.. code-block:: python

    from illuma_samc import SAMCConfig

    cfg = SAMCConfig(n_bins=40, e_min=0, e_max=10, gain="1/t", gain_t0=1000)
    wm = cfg.build()          # SAMCWeights
    sampler = cfg.build_sampler(energy_fn=my_energy, dim=2)  # full SAMC

Load from YAML:

.. code-block:: python

    cfg = SAMCConfig.from_yaml("configs/samc.yaml", model="2d")
