from pathlib import Path

from PIL import Image
import torch
import torchvision.transforms.functional as TF


def preprocess_image(path) -> torch.Tensor:
    """
    Exact historical EviRoute preprocessing.

    RGB
      -> direct 224x224 bicubic resize
      -> ToTensor
      -> float [0,1]

    No crop.
    No padding.
    No RGB mean/std normalization.
    """

    path = Path(path)

    img = Image.open(path).convert("RGB")

    # Preserve historical implementation.
    img = img.resize(
        (224, 224),
        Image.BICUBIC,
    )

    tensor = TF.to_tensor(img)

    if tensor.shape != (3, 224, 224):
        raise RuntimeError(
            f"Unexpected image tensor shape: "
            f"{tuple(tensor.shape)}"
        )

    return tensor.unsqueeze(0)
