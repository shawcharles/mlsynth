Generalized Synthetic Control (GSC)
===================================

The Generalized Synthetic Control (GSC) method is often associated with denoising matrix completion techniques for estimating treatment effects in panel data settings. This implementation is guided by the approach described in Costa et al. (2023), which utilizes a denoising algorithm to estimate counterfactual outcomes.

For a detailed theoretical background and methodology, please refer to:

- Costa, L., Farias, V. F., Foncea, P., Gan, J. (D.), Garg, A., Montenegro, I. R., Pathak, K., Peng, T., & Popovic, D. (2023). "Generalized Synthetic Control for TestOps at ABI: Models, Algorithms, and Infrastructure." *INFORMS Journal on Applied Analytics* 53(5):336-349.

Estimator API
-------------

.. automodule:: mlsynth.estimators.gsc
   :no-members:
   :no-inherited-members:

.. currentmodule:: mlsynth.estimators.gsc

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   GSC

Class Reference
---------------

.. autoclass:: GSC
   :members:
   :inherited-members:
   :undoc-members:
   :show-inheritance:

Configuration Class Reference
-----------------------------

.. currentmodule:: mlsynth.config_models

.. autoclass:: GSCConfig
   :members:
   :inherited-members:
   :show-inheritance:
