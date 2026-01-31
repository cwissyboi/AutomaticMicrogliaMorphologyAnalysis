from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import cv2
from random_disconnections.disconnect_components import disconnect_branches_with_gap
from random_disconnections.find_connection_points import find_random_branch_points
from random_disconnections.random_colour_noise import add_floating_synthetic_fragments



class SegmentationDataset(Dataset):
    def __init__(self, df, transform=None, disconnect_components = True, add_new_components = True):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.disconnect_components = disconnect_components
        self.add_new_components = add_new_components

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.loc[idx, "image_path"]
        mask_path = self.df.loc[idx, "mask_path"]

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = np.array(image)
        mask = np.array(mask)

        # Normalize mask to {0,1}
        mask = (mask > 0).astype(np.float32)

        # Override the image variables above if you want random changes
        if (self.disconnect_components): 
            
            points = find_random_branch_points(
                image_path=img_path, 
                mask_path = mask_path, 
                points_per_branch=5, 
                min_branch_length=10
            )

            H, W = image.shape[:2]
            box_size = int(0.1 * H)   # 10% of image height

            # --- cut branches locally ---
            image, _ = disconnect_branches_with_gap(
                img_path,
                mask_path,
                points, 
                box_size = box_size,
                blur_output = True, 
                blur_feather_radius=1
            )
            
            # swap order of colours to make it consitent with the code below
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.add_new_components: 
            image = add_floating_synthetic_fragments(
                    image,
                    mask
                )


        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        image = image.float() / 255.0
        mask = mask.unsqueeze(0)  # (1, H, W)

        return image, mask