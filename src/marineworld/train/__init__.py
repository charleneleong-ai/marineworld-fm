"""Training entrypoints.

v1 adds a Lightning module for VideoMAE-style masked pretraining plus a
few-shot fine-tuning entrypoint. Every entrypoint must call
`marineworld.utils.seed.seed_everything()` before constructing models/data.
"""
