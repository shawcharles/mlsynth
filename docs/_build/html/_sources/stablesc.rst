Stable Synthetic Control (StableSC)
===================================

The Stable Synthetic Control (StableSC) method enhances traditional synthetic control by employing a hybrid anomaly-based donor selection process. It identifies suitable donor units by combining insights from Granger causality tests and proximity measures (based on chi-squared distances of time series). This approach aims to construct a more robust and stable synthetic counterfactual for the treated unit.

The core idea is to filter potential donors first by their Granger causality with the treated unit's outcome and their proximity to the average of other donors. Then, Radial Basis Function (RBF) scores are computed based on proximity distances to weigh the selected donors. Finally, an optimization is performed to find the synthetic control weights.

Estimator API
-------------

.. automodule:: mlsynth.estimators.stablesc
   :no-members:
   :no-inherited-members:

.. currentmodule:: mlsynth.estimators.stablesc

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   StableSC

Class Reference
---------------

.. autoclass:: StableSC
   :members:
   :inherited-members:
   :undoc-members:
   :show-inheritance:

Configuration Class Reference
-----------------------------

.. currentmodule:: mlsynth.config_models

.. autoclass:: StableSCConfig
   :members:
   :inherited-members:
   :show-inheritance:
