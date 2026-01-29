# import albumentations as A
# import numpy as np

# class MyTestTransform(A.DualTransform):
#     def __init__(self, p=0.5):
#         super().__init__(p=p)

#     def apply(self, image, mask_for_apply=None, **params):
#         print("bruh")
#         print("mask shape:", mask_for_apply.shape)
#         return 255 - image

#     def apply_to_mask(self, mask, **params):
#         return mask

#     def get_params_dependent_on_targets(self, params):
#         image = params["image"]
#         mask  = params["mask"]

#         # need to pass it like this into the apply function so that it has access to the mask as well as the image
#         return {
#             "mask_for_apply": mask
#         }

#     @property
#     def targets_as_params(self):
#         return ["image", "mask"]


import albumentations as A

class MyTestTransform(A.DualTransform):
    def __init__(self, p=1.0):
        super().__init__(p=p)

    def get_params_dependent_on_targets(self, params):
        image = params["image"]
        mask  = params["mask"]

        # joint logic here
        invert = mask.sum() > 0

        return {"invert": invert}

    def apply(self, image, invert=False, **params):
        if invert:
            return 255 - image
        return image

    def apply_to_mask(self, mask, **params):
        return mask

    @property
    def targets_as_params(self):
        return ["image", "mask"]

