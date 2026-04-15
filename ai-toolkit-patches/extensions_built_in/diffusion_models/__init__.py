# Only import the models we actually use for Klein LoRA training.
# The full ai-toolkit __init__ imports every model (Chroma, HiDream, Wan2, etc.)
# which drags in heavy deps (T5EncoderModel, UMT5, etc.) that we don't need.
from .flux2 import Flux2Model, Flux2KleinModel

AI_TOOLKIT_MODELS = [
    Flux2Model,
    Flux2KleinModel,
]
