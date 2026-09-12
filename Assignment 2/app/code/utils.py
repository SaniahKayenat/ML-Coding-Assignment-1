"""
Small persistence helpers.

We use joblib rather than raw pickle because our saved object is a whole
scikit-learn Pipeline and joblib is what scikit-learn's own docs
recommend for this
"""

import joblib


def save(filename: str, obj: object):
    joblib.dump(obj, filename)


def load(filename: str) -> object:
    return joblib.load(filename)
