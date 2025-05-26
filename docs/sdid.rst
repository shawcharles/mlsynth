Synthetic Difference-in-Differences (SDID)
===========================================

The Synthetic Difference-in-Differences (SDID) method combines synthetic control weighting for units and time periods with a difference-in-differences estimation framework. It is designed for panel data settings, typically with a single treated unit (or a single treatment adoption time across multiple units treated simultaneously) and multiple control units, observed over pre- and post-treatment periods.

For a detailed theoretical background and methodology, please refer to:

- Arkhangelsky, D., Athey, S., Hirshberg, D. A., Imbens, G. W., & Wager, S. (2021). "Synthetic Difference-in-Differences." *American Economic Review*, 111(12), 4088-4118.
- Athey, S., Bayati, M., Doudchenko, N., Imbens, G., & Khosravi, K. (2021). "Matrix Completion Methods for Causal Panel Data Models." *Journal of the American Statistical Association*, 116(536), 1716-1730. (Though this paper covers broader matrix completion, SDID is related).

Estimator API
-------------

.. automodule:: mlsynth.estimators.sdid
   :no-members:
   :no-inherited-members:

.. currentmodule:: mlsynth.estimators.sdid

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   SDID

Class Reference
---------------

.. autoclass:: SDID
   :members:
   :inherited-members:
   :undoc-members:
   :show-inheritance:

Configuration Class Reference
-----------------------------

.. currentmodule:: mlsynth.config_models

.. autoclass:: SDIDConfig
   :members:
   :inherited-members:
   :show-inheritance:
