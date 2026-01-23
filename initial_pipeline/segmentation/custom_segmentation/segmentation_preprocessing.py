
import cv2
from skimage.graph import route_through_array
from scipy.ndimage import distance_transform_edt
from data_utils import index_segmentations_df, export_calculated_masks
from plotting_utils import show_row_visuals, show_mask_outline
from crf import add_calculated_mask_column
from pathlib import Path



def preprocess_segmentations(): 
    SEGMENTATIONS_DIR = Path.cwd().parents[2] / "AnnotationsData" / "Segmentations"

    df = index_segmentations_df(SEGMENTATIONS_DIR)

    # Only include the whole cell annotation for now, later we can do a custom soma segmentation
    df = df[df['class'] == 'MG_whole']
    print(len(df), "annotation pairs found")
    df.head()

    print(df.columns)



    df = add_calculated_mask_column(df)


    bad_image_quality = [
    4, 11, 15, 60, 92, 191, 193, 206, 218, 247,
    329, 393, 455, 457, 462, 532, 794
]

    bad = [
        0, 38, 42, 44, 82, 176, 261, 285, 345, 380,
        381, 395, 446, 448, 451, 474, 478, 479,
        483, 490
    ]

    medium = [
        3, 8, 13, 16, 19, 23, 39, 46, 47, 48, 49,
        73, 80, 91, 101, 109, 116, 132, 137, 144,
        148, 149, 159, 160, 162, 165, 170, 171,
        172, 174, 175, 177, 182, 183, 184, 185,
        186, 187, 188, 190, 196, 197, 198, 199,
        200, 201, 203, 204, 205, 207, 208, 209,
        210, 211, 216, 217, 219, 220, 221, 222,
        223, 224, 226, 228, 229, 231, 232, 233,
        234, 235, 239, 241, 242, 245, 250, 259,
        260, 262, 264, 265, 268, 269, 271, 272,
        274, 276, 281, 283, 284, 286, 287, 288,
        293, 294, 295, 296, 297, 299, 300, 304,
        305, 306, 308, 309, 310, 311, 312, 314,
        315, 318, 319, 323, 324, 325, 326, 327,
        328, 330, 331, 332, 333, 336, 337, 339,
        340, 341, 343, 348, 351, 355, 359, 361,
        364, 369, 371, 373, 376, 377, 378, 386,
        396, 397, 401, 404, 405, 407, 408, 410,
        412, 414, 415, 420, 423, 427, 428, 430,
        434, 437, 439, 440, 441, 452, 453, 456,
        460, 461, 464, 472, 473, 476, 477, 480,
        482, 486, 488, 489, 492, 495, 496, 497,
        498, 501, 508, 511, 514, 515, 516, 520,
        528, 529, 531, 534, 536, 539, 541, 545,
        548, 550, 554, 557, 559, 560, 562, 564,
        565, 573, 578, 581, 584, 585, 593, 598,
        605, 609, 611, 614, 619, 622, 625, 628,
        629, 635, 636, 645, 647, 653, 655, 656,
        658, 659, 662, 666, 667, 671, 674, 675,
        679, 680, 685, 688, 690, 700, 701, 702,
        704, 708, 712, 715, 718, 721, 729, 730,
        732, 737, 739, 740, 747, 757, 765, 770,
        773, 781, 787, 791, 805, 806, 809, 811,
        819, 821, 826, 830, 844, 845, 850, 855,
        856, 857, 866, 867, 876, 878, 881, 888,
        895, 896, 898, 902, 908, 909, 911, 925,
        931, 932, 938, 945, 948, 952, 956, 963,
        970, 972, 974, 977, 978, 986, 987, 995,
        996, 997, 999, 1004, 1006, 1008
    ]


    too_big = [18, 21, 25, 33, 36, 37, 72, 99, 416]

    disagree = [77, 105, 344, 383, 497]


    # Start with default
    df["mask_quality"] = "good"

    # Assign based on index membership
    # df.loc[df.index.isin(bad_image_quality), "mask_quality"] = "bad_image_quality"
    # df.loc[df.index.isin(bad), "mask_quality"] = "bad"
    # df.loc[df.index.isin(medium), "mask_quality"] = "medium"
    # df.loc[df.index.isin(too_big), "mask_quality"] = "too_big"
    # df.loc[df.index.isin(disagree), "mask_quality"] = "disagree"

    df["mask_quality"] = "good"

    df.iloc[bad_image_quality, df.columns.get_loc("mask_quality")] = "bad_image_quality"
    df.iloc[bad, df.columns.get_loc("mask_quality")] = "bad"
    df.iloc[medium, df.columns.get_loc("mask_quality")] = "medium"
    df.iloc[too_big, df.columns.get_loc("mask_quality")] = "too_big"
    df.iloc[disagree, df.columns.get_loc("mask_quality")] = "disagree"


    df_to_output = df[['image_path', 'mask_path', 'class', 'scan', 'mask_quality']]
    df_to_output.to_csv('mask_quality_summary.csv')


  
    export_calculated_masks(df, SEGMENTATIONS_DIR, mask_column='calculated_mask')


if __name__ == "__main__":
    preprocess_segmentations()