"""
driver_state
============
Shared building blocks for training the final models and running live inference.

Modules that import torch (models, device, inference) are kept separate from the
numpy-only modules (config, calibration, data, sensors, display) so that the
sensor simulator processes start quickly and never touch the GPU.
"""
