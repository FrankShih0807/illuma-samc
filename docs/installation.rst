Installation
============

This page covers how to install ``illuma-samc`` from source, set up a development
environment, and verify that the package is working correctly.

Prerequisites
-------------

**Python**

``illuma-samc`` requires **Python 3.11 or later**.
Check your version with:

.. code-block:: bash

   python --version

**PyTorch**

The only mandatory runtime dependency is **PyTorch >= 2.0**. PyTorch is *not*
installed automatically because the correct variant (CPU-only, CUDA 11.x, CUDA 12.x,
ROCm, etc.) depends on your hardware.

Visit `pytorch.org <https://pytorch.org/get-started/locally/>`_ and use the
interactive selector to get the right install command for your system.  A typical
CUDA 12 install looks like:

.. code-block:: bash

   pip install torch --index-url https://download.pytorch.org/whl/cu124

.. note::

   GPU support is optional. ``illuma-samc`` runs on CPU by default and accepts a
   ``device`` parameter wherever tensors are created, so you can switch to
   ``"cuda"`` without changing algorithm code.

Conda Environment (Recommended)
---------------------------------

Using ``conda`` is strongly recommended because it manages CUDA libraries
alongside Python packages, avoiding version conflicts.

Create and activate a dedicated environment:

.. code-block:: bash

   conda create -n illuma-samc python=3.11
   conda activate illuma-samc

Install PyTorch inside the environment (select the command from
`pytorch.org <https://pytorch.org/get-started/locally/>`_ that matches your
hardware), then proceed to the package install below.

.. note::

   All subsequent commands assume the ``illuma-samc`` conda environment is
   active.  If you see import errors, double-check with ``conda info --envs``
   that the right environment is activated.

Install from Source
-------------------

Clone the repository and install in editable mode with all development
dependencies:

.. code-block:: bash

   git clone https://github.com/FrankShih0807/illuma-samc.git
   cd illuma-samc
   pip install -e ".[dev]"

The ``[dev]`` extras install everything needed for testing, linting, and building
these docs.  See :ref:`optional-extras` below for a breakdown of what each
extra group contains.

.. _optional-extras:

Optional Extras
---------------

``pyproject.toml`` defines several optional dependency groups:

.. list-table::
   :header-rows: 1
   :widths: 15 55 30

   * - Extra
     - What it adds
     - Install command
   * - ``viz``
     - ``matplotlib >= 3.7`` — used by plotting helpers
     - ``pip install -e ".[viz]"``
   * - ``progress``
     - ``tqdm >= 4.66`` — progress bars during sampling runs
     - ``pip install -e ".[progress]"``
   * - ``full``
     - Both ``viz`` and ``progress``
     - ``pip install -e ".[full]"``
   * - ``docs``
     - ``sphinx``, ``sphinx-rtd-theme``, ``sphinx-autodoc-typehints``
     - ``pip install -e ".[docs]"``
   * - ``dev``
     - Everything above plus ``pytest``, ``pytest-cov``, ``ruff``, ``pyyaml``
     - ``pip install -e ".[dev]"``

Development Dependencies
------------------------

The ``[dev]`` group installs the full toolchain used in this project:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Tool
     - Purpose
   * - ``pytest >= 8.0``
     - Test runner — execute with ``pytest`` from the repo root
   * - ``pytest-cov >= 5.0``
     - Coverage reporting alongside pytest
   * - ``ruff >= 0.4``
     - Formatter (``ruff format .``) and linter (``ruff check .``)
   * - ``matplotlib >= 3.7``
     - Plotting in examples and benchmark scripts
   * - ``tqdm >= 4.66``
     - Progress bars
   * - ``sphinx >= 7.0``
     - Documentation builder (this site)
   * - ``sphinx-rtd-theme >= 2.0``
     - Read-the-Docs HTML theme
   * - ``sphinx-autodoc-typehints >= 2.0``
     - Renders type hints in API reference pages

Verifying the Installation
--------------------------

After installation, confirm the package imports correctly and check the version:

.. code-block:: python

   import illuma_samc
   print(illuma_samc.__version__)

For a slightly more thorough smoke test, run a minimal sampler on a toy energy
function:

.. code-block:: python

   import torch
   from illuma_samc import SAMC

   def energy_fn(x):
       return torch.min(
           0.5 * torch.sum((x - 2) ** 2),
           0.5 * torch.sum((x + 2) ** 2),
       )

   sampler = SAMC(energy_fn=energy_fn, dim=2)
   result = sampler.run(n_steps=10_000)

   print(f"Best energy:     {result.best_energy:.4f}")
   print(f"Acceptance rate: {result.acceptance_rate:.3f}")

If both lines print without error, the install is working. Expected output:

.. code-block:: text

   Best energy:     ~0.0000
   Acceptance rate: ~0.3–0.6

.. note::

   Exact acceptance rate varies by random seed and step count; values in the
   range 0.2–0.6 are normal for this toy problem.

Running the Test Suite
----------------------

From the repository root with the conda environment active:

.. code-block:: bash

   pytest

To include a coverage report:

.. code-block:: bash

   pytest --cov=illuma_samc --cov-report=term-missing

.. warning::

   A small number of tests exercise GPU paths and will be skipped automatically
   when no CUDA device is detected.  This is expected on CPU-only machines.

Next Steps
----------

- :doc:`quickstart` — a complete end-to-end example with plots.
- :doc:`api` — full API reference auto-generated from docstrings.
