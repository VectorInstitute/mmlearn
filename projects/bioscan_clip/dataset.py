import io
from typing import Literal, Union, Optional
import h5py
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from mmlearn.datasets.core import Example, Modalities
from mmlearn.constants import EXAMPLE_INDEX_KEY

from projects.bioscan_clip.dna_tokenizer import DNAProcessor

# NOTE: this module must be imported before any use of Modalities.DNA anywhere
# else in the code
Modalities.register_modality("dna")


class BIOSCANInsectDataset(Dataset[Example]):
    def __init__(
        self,
        path_to_hdf5_file: str,
        variant: Literal["1m", "5m"],
        split: Literal[
            "all_keys",
            "no_split",
            "no_split_and_seen_train",
            "seen_keys",
            "single_species",
            "test_seen",
            "test_unseen",
            "test_unseen_keys",
            "train_seen",
            "val_seen",
            "val_unseen",
            "val_unseen_keys",
            "unseen_keys",
        ],
        path_to_tsv_data: Optional[str] = None,
        image_input_type: Literal["image", "feature"] = "image",
        dna_inout_type: Literal["sequence", "feature"] = "sequence",
        dna_processor: Optional[DNAProcessor] = None,
        labels: Optional[Union[int, list[int]]] = None,
        for_training: bool = True,
        bin_for_positive_and_negative_pairs: bool = False,
    ) -> None:
        if dna_inout_type not in ["sequence", "feature"]:
            raise TypeError(
                "Expected `dna_inout_type` to be either 'sequence' or 'feature', "
                f"but got {dna_inout_type}. Please check the configuration."
            )

        self.path_to_hdf5_file = path_to_hdf5_file
        self.variant = variant
        self.split = split
        self.path_to_hdf5_file = path_to_hdf5_file
        self.image_input_type = image_input_type
        self.dna_inout_type = dna_inout_type
        self.dna_processor = dna_processor
        self.for_training = for_training
        self.bin_for_positive_and_negative_pairs = bin_for_positive_and_negative_pairs

        self.data = h5py.File(path_to_hdf5_file, "r", libver="latest")[split]

        if self.dna_processor is None:
            self.dna_processor = DNAProcessor(max_length=660)

        self.transform = None
        if self.image_input_type == "image":
            if self.for_training:
                self.transform = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Resize(size=256, antialias=True),
                        transforms.RandomResizedCrop(224, antialias=True),
                        transforms.Normalize(
                            (0.48145466, 0.4578275, 0.40821073),
                            (0.26862954, 0.26130258, 0.27577711),
                        ),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomVerticalFlip(),
                        transforms.RandomRotation(degrees=(-45, 45)),
                    ]
                )
            else:
                self.transform = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Resize(size=256, antialias=True),
                        transforms.CenterCrop(224),
                        transforms.Normalize(
                            (0.48145466, 0.4578275, 0.40821073),
                            (0.26862954, 0.26130258, 0.27577711),
                        ),
                    ]
                )
        elif self.image_input_type != "feature":
            raise TypeError(
                "Expected `image_input_type` to be either 'image' or 'feature', "
                f"but got {image_input_type}. Please check the configuration."
            )

        list_of_label_dict = get_array_of_label_dicts(path_to_hdf5_file, split)
        self.list_of_label_string = []
        for i in list_of_label_dict:
            self.list_of_label_string.append(
                i["order"] + " " + i["family"] + " " + i["genus"] + " " + i["species"]
            )

        if self.for_training:
            if bin_for_positive_and_negative_pairs:
                self.labels = list(
                    get_bin_from_tsv(split, path_to_hdf5_file, path_to_tsv_data)
                )
                self.labels = np.array(convert_uri_to_index_list(self.labels))
            elif labels is None:
                self.labels = np.array(list(range(len(self.data["image"]))))
            else:
                self.labels = labels
        else:
            self.labels = get_array_of_label_dicts(path_to_hdf5_file, split)

    def __len__(self) -> int:
        return len(self.data["image"])

    def load_image(self, idx: int) -> Image:
        image_enc_padded = self.data["image"][idx].astype(np.uint8)
        enc_length = self.data["image_mask"][idx]
        image_enc = image_enc_padded[:enc_length]
        image = Image.open(io.BytesIO(image_enc))

        if self.transform is not None:
            image = self.transform(image)

        return image

    def __getitem__(self, idx: int) -> Example:
        if self.image_input_type == "image":
            image = self.load_image(idx)
        else:
            image = self.data["image_features"][idx].astype(np.float32)

        if self.dna_inout_type == "sequence":
            dna_seq = self.dna_processor(self.data["barcode"][idx].decode("utf-8"))
        else:
            dna_seq = self.data["dna_features"][idx].astype(np.float32)

        if self.variant == "5m":
            curr_processid = self.data["processid"][idx].decode("utf-8")
        else:
            curr_processid = self.data["image_file"][idx].decode("utf-8")

        language_input_ids = self.data["language_tokens_input_ids"][idx]
        language_token_type_ids = self.data["language_tokens_token_type_ids"][idx]
        language_attention_mask = self.data["language_tokens_attention_mask"][idx]

        return Example(
            {
                EXAMPLE_INDEX_KEY: idx,
                Modalities.RGB: image,
                Modalities.DNA: dna_seq,
                Modalities.TEXT: language_input_ids,
                "language_token_type_ids": language_token_type_ids,
                Modalities.TEXT.attention_mask: language_attention_mask,
                "labels": self.labels[idx],
                "process_id": curr_processid,
                "split": self.split,
            }
        )


def get_array_of_label_dicts(hdf5_inputs_path: str, split: str) -> np.array:
    hdf5_split_group = h5py.File(hdf5_inputs_path, "r", libver="latest")[split]
    np_order = np.array([item.decode("utf-8") for item in hdf5_split_group["order"][:]])
    np_family = np.array(
        [item.decode("utf-8") for item in hdf5_split_group["family"][:]]
    )
    np_genus = np.array([item.decode("utf-8") for item in hdf5_split_group["genus"][:]])
    np_species = np.array(
        [item.decode("utf-8") for item in hdf5_split_group["species"][:]]
    )
    return np.array(
        [
            {"order": o, "family": f, "genus": g, "species": s}
            for o, f, g, s in zip(np_order, np_family, np_genus, np_species)
        ],
        dtype=object,
    )


def get_bin_from_tsv(split: str, hdf5_path: str, tsv_path: str) -> list[str]:
    with h5py.File(hdf5_path, "r") as h5file:
        sample_id_list = [item.decode("utf-8") for item in h5file[split]["sampleid"]]
    df = pd.read_csv(tsv_path, sep="\t")
    filtered_df = df[df["sampleid"].isin(sample_id_list)]
    return filtered_df["uri"].tolist()


def convert_uri_to_index_list(uri_list: list[str]) -> list[int]:
    string_to_int = {}
    next_int = 0
    integers = []
    for s in uri_list:
        if s not in string_to_int:
            string_to_int[s] = next_int
            next_int += 1
        integers.append(string_to_int[s])

    return integers
