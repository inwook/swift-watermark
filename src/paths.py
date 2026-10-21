import os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def OUT_PATH(*parts):
    p = os.path.join(_PROJECT_ROOT, "output", *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p

def TMP_PATH(*parts):
    p = os.path.join(_PROJECT_ROOT, "tmp", *parts)
    os.makedirs(os.path.dirname(p) if parts else p, exist_ok=True)
    return p

def CACHE_PATH(*parts):
    p = os.path.join(_PROJECT_ROOT, "cache", *parts)
    os.makedirs(os.path.dirname(p) if parts else p, exist_ok=True)
    return p

def TESTPROJECTS_PATH(*parts):
    p = os.path.join(_PROJECT_ROOT, "testprojects", *parts)
    return p

def BASELINE_PATH(*parts):
    return os.path.join(_PROJECT_ROOT, "baseline_outputs", *parts)
