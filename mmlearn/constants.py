"""Constants."""

from typing import Callable, Dict, List, Tuple


EXAMPLE_INDEX_KEY = "example_index"

data = {
    "VinDrMammo": [
        "a x-ray image showing {c}",
        "mammography image of {c}",
        "mammogram showing {c}",
        "presence of {c} on mammogram",
    ],
    "HAM10000": [
        "a histopathology slide showing {c}",
        "histopathology image of {c}",
        "pathology tissue showing {c}",
        "presence of {c} tissue on image",
    ],
    "PadUfes20": [
        "a histopathology slide showing {c}",
        "histopathology image of {c}",
        "pathology tissue showing {c}",
        "presence of {c} tissue on image",
    ],
    "pathmnist": [
        "adipose",
        "background",
        "debris",
        "lymphocytes",
        "mucus",
        "smooth muscle",
        "normal colon mucosa",
        "cancer-associated stroma",
        "colorectal adenocarcinoma epithelium",
    ],
    "chestmnist": [
        "atelectasis",
        "cardiomegaly",
        "effusion",
        "infiltration",
        "mass",
        "nodule",
        "pneumonia",
        "pneumothorax",
        "consolidation",
        "edema",
        "emphysema",
        "fibrosis",
        "pleural",
        "hernia",
    ],
    "dermamnist": [
        "actinic keratoses and intraepithelial carcinoma",
        "basal cell carcinoma",
        "benign keratosis-like lesions",
        "dermatofibroma",
        "melanoma",
        "melanocytic nevi",
        "vascular lesions",
    ],
    "octmnist": [
        "choroidal neovascularization",
        "diabetic macular edema",
        "drusen",
        "normal",
    ],
    "pneumoniamnist": ["normal", "pneumonia"],
    "retinamnist": [
        "no apparent retinopathy",
        "mild NPDR, non-proliferative diabetic retinopathy",
        "moderate NPDR, non-proliferative diabetic retinopathy",
        "severe NPDR, non-proliferative diabetic retinopathy",
        "PDR, proliferative diabetic retinopathy",
    ],
    "breastmnist": ["malignant", "normal, benign"],
    "bloodmnist": [
        "basophil",
        "eosinophil",
        "erythroblast",
        "immature granulocytes(myelocytes, metamyelocytes and promyelocytes)",
        "lymphocyte",
        "monocyte",
        "neutrophil",
        "platelet",
    ],
    "tissuemnist": [
        "Collecting Duct, Connecting Tubule",
        "Distal Convoluted Tubule",
        "Glomerular endothelial cells",
        "Interstitial endothelial cells",
        "Leukocytes",
        "Podocytes",
        "Proximal Tubule Segments",
        "Thick Ascending Limb",
    ],
    "organamnist": [
        "bladder",
        "femur-left",
        "femur-right",
        "heart",
        "kidney-left",
        "kidney-right",
        "liver",
        "lung-left",
        "lung-right",
        "pancreas",
        "spleen",
    ],
    "organcmnist": [
        "bladder",
        "femur-left",
        "femur-right",
        "heart",
        "kidney-left",
        "kidney-right",
        "liver",
        "lung-left",
        "lung-right",
        "pancreas",
        "spleen",
    ],
    "organsmnist": [
        "bladder",
        "femur-left",
        "femur-right",
        "heart",
        "kidney-left",
        "kidney-right",
        "liver",
        "lung-left",
        "lung-right",
        "pancreas",
        "spleen",
    ],
    "lc25000_colon": [
        "a histopathology slide showing {c}",
        "histopathology image of {c}",
        "pathology tissue showing {c}",
        "presence of {c} tissue on image",
    ],
    "lc25000_lung": [
        "a histopathology slide showing {c}",
        "histopathology image of {c}",
        "pathology tissue showing {c}",
        "presence of {c} tissue on image",
    ],
    "sicap": [
        "a histopathology slide showing {c}",
        "histopathology image of {c}",
        "pathology tissue showing {c}",
        "presence of {c} tissue on image",
    ],
}


def create_lambda(t: str) -> Callable[[str], str]:
    """Create a lambda that replaces {c} in the template."""
    return lambda c: t.replace("{c}", str(c))


def convert_to_lambda(templates: List[str]) -> Tuple[Callable[[str], str], ...]:
    """
    Convert a list of template strings into a tuple of lambda functions.

    Each lambda will accept a variable `c` and replace '{c}' in the template
    with the value of `c`.

    Args:
        templates (List[str]): A list of template strings containing '{c}'.

    Returns
    -------
        Tuple[Callable[[str], str], ...]: A tuple of lambdas that replace '{c}'
        with a given value.
    """
    return tuple(create_lambda(t) for t in templates)


TEMPLATES: Dict[str, Tuple[Callable[[str], str], ...]] = {
    key: convert_to_lambda(value) for key, value in data.items()
}
