import albumentations as A
import numpy as np

class MyTestTransform(A.DualTransform):
    def __init__(self, p=0.5):
        super().__init__(p=p)

    def apply(self, image, **params):
        # image: (H, W, 3), uint8
        print("bruh")  # just to confirm it runs
        return 255 - image

    def apply_to_mask(self, mask, **params):
        # masks should NOT be inverted
        return mask

    def get_params_dependent_on_targets(self, params):
        # You don't actually need this for inversion,
        # but keeping it since you asked about it earlier
        image = params["image"]
        mask  = params["mask"]

        value = image.mean() + mask.mean()
        return {"value": value}

    @property
    def targets_as_params(self):
        return ["image", "mask"]
