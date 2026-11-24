"""Compatibility exports (ARE-009C); retain through ARE-009D/E caller migration.

New code uses artifact_io/contracts. save_models is now deliberately immutable.
No model is rewritten on load or for legacy format adaptation.
"""
from anime_recommender.artifact_io import load_models, mapping_fingerprint, save_models
from anime_recommender.inference.bundle import ModelBundle
