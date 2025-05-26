Estimation Utilities (`mlsynth.utils.estutils`)
================================================

This module provides a collection of utility functions supporting various estimation procedures within the `mlsynth` library. These include optimization routines for Synthetic Control Methods (SCM), cross-validation helpers, inference utilities, and specific algorithms for methods like Panel Data Approach (PDA) and Proximal Inference.

Class Reference
---------------

.. currentmodule:: mlsynth.utils.estutils

.. autoclass:: Opt
   :members: SCopt
   :undoc-members:
   :show-inheritance:

Function Reference
------------------

.. currentmodule:: mlsynth.utils.estutils

.. autofunction:: pi2
.. autofunction:: compute_hac_variance
.. autofunction:: compute_t_stat_and_ci
.. autofunction:: l2_relax
.. autofunction:: cross_validate_tau
.. autofunction:: ci_bootstrap
.. autofunction:: TSEST
.. autofunction:: pda
.. autofunction:: bartlett
.. autofunction:: hac
.. autofunction:: pi
.. autofunction:: pi_surrogate
.. autofunction:: pi_surrogate_post
.. autofunction:: get_theta
.. autofunction:: get_sigmasq
.. autofunction:: SRCest
.. autofunction:: RPCASYNTH
.. autofunction:: SMOweights
.. autofunction:: NSC_opt
.. autofunction:: NSCcv
.. autofunction:: pcr

Internal Helper Functions
-------------------------
These functions are primarily used internally by other functions in this module or by estimators.

.. autofunction:: _estimate_single_sc_model_for_tsest
.. autofunction:: _solve_pda_fs
.. autofunction:: _solve_pda_lasso
.. autofunction:: _solve_pda_l2
.. autofunction:: __SRC_opt
