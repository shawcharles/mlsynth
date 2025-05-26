Nonlinear Synthetic Control (NSC)
===================================

The Nonlinear Synthetic Control (NSC) method estimates the treatment effect for a single treated unit using an affine combination of control units. The method involves a regularization term that encourages sparsity and stability in the weights. Hyperparameters (denoted `a` and `b` in the underlying optimization) for the weight estimation are selected via cross-validation (`NSCcv`) to minimize pre-treatment Mean Squared Prediction Error (MSPE).

Estimator API
-------------

.. automodule:: mlsynth.estimators.nsc
   :no-members:
   :no-inherited-members:

.. currentmodule:: mlsynth.estimators.nsc

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   NSC

Class Reference
---------------

.. autoclass:: NSC
   :members:
   :inherited-members:
   :undoc-members:
   :show-inheritance:

Configuration Class Reference
-----------------------------

.. currentmodule:: mlsynth.config_models

.. autoclass:: NSCConfig
   :members:
   :inherited-members:
   :show-inheritance:
