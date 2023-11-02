# enhancements/__init__.py

# Import necessary modules and subpackages for convenience
from .memory import (
    compute_fisher_information,
    estimate_model_size,
    estimate_data_size,
    approximate_hessian,
    prune_model,
    quantize_model,
    distill_knowledge,
    compute_activation_statistics,
    visualize_weight_sparsity,
    profile_memory,
    clip_activations,
    detect_memory_leak,
    estimate_serialized_model_size,
    normalize_data,
    save_model_checkpoint,
    clear_memory,
    suggest_memory_optimizations
)

from .meta_learning import (
    SelfModifyingAI,
    self_adjusting_hyperparameter_tuning
)

# Optionally, you can define package-level variables, functions, or classes here.

__version__ = "1.0.0"
