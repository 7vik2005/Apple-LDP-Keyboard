"""
HCMS Estimator — Frequency Estimation for Hadamard Count Mean Sketch.

After the server calls HCMSServer.finalize(), the reconstructed sketch matrix M
has the same shape and interpretation as the CMS sketch matrix. Frequency
estimation therefore re-uses the shared CMS estimator from src.cms.estimator.

This module is intentionally minimal: it documents the design decision and
ensures the package is importable as src.hcms.estimator if needed.

See Also:
    src.cms.estimator.estimate_frequency   — shared unbiased estimator
    src.hcms.server.HCMSServer.finalize   — FWHT reconstruction step
"""
