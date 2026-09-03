"""
Preprocessing package for DR-EarlyXAI.
"""

def __getattr__(name):
    if name == "FundusPreprocessor":
        from .fundus_preprocessing import FundusPreprocessor
        return FundusPreprocessor
    elif name == "QualityChecker":
        from .quality_check import QualityChecker
        return QualityChecker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def __dir__():
    return ["FundusPreprocessor", "QualityChecker"]
