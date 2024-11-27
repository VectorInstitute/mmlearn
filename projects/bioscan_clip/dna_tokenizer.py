import itertools

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast


class DNAProcessor:
    def __init__(self, max_length: int, kmers: int = 5, stride: int = 5) -> None:
        self.tokenizer = create_dna_tokenizer(kmers)
        self.padder = PadDNASequence(max_length)
        self.kmer_processor = KmerProcessor(kmers, stride)

    def __call__(self, dna_sequence: str) -> list[int]:
        kmers = self.kmer_processor(self.padder(dna_sequence))
        return self.tokenizer.encode(
            kmers, is_split_into_words=True, return_tensors="pt"
        )[0]


def create_dna_tokenizer(k: int = 5) -> PreTrainedTokenizerFast:
    special_tokens = ["[MASK]", "[CLS]", "[UNK]"]
    kmer_iter = ("".join(kmer) for kmer in itertools.product("ACGT", repeat=k))

    tokenizer = Tokenizer(
        WordLevel(
            vocab={
                token: i
                for i, token in enumerate(itertools.chain(special_tokens, kmer_iter))
            },
            unk_token="[UNK]",
        )
    )
    tokenizer.add_special_tokens(special_tokens)
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A",
        pair="[CLS] $A $B:1",
        special_tokens=[
            ("[CLS]", tokenizer.token_to_id("[CLS]")),
            ("[UNK]", tokenizer.token_to_id("[UNK]")),
            ("[MASK]", tokenizer.token_to_id("[MASK]")),
        ],
    )

    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        mask_token="[MASK]",
        cls_token="[CLS]",
    )


class PadDNASequence:
    def __init__(self, max_len: int) -> None:
        self.max_len = max_len

    def __call__(self, dna_sequence: str) -> str:
        if len(dna_sequence) > self.max_len:
            return dna_sequence[: self.max_len]
        return dna_sequence + "N" * (self.max_len - len(dna_sequence))


class KmerProcessor:
    def __init__(self, k: int, stride: int = 1) -> None:
        self.k = k
        self.stride = stride

    def __call__(self, dna_sequence: str) -> list[str]:
        tokens = []
        for i in range(0, len(dna_sequence) - self.k + 1, self.stride):
            k_mer = dna_sequence[i : i + self.k]
            tokens.append(k_mer)
        return tokens
