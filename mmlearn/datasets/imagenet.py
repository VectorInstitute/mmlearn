"""ImageNet dataset."""

import os
from typing import Any, Callable, Literal, Optional

from hydra_zen import MISSING, store
from torchvision.datasets.folder import ImageFolder

from mmlearn.constants import EXAMPLE_INDEX_KEY
from mmlearn.datasets.core import Modalities
from mmlearn.datasets.core.example import Example


@store(
    group="datasets",
    provider="mmlearn",
    root_dir=os.getenv("IMAGENET_ROOT_DIR", MISSING),
)
class ImageNet(ImageFolder):
    """ImageNet dataset.

    This is a wrapper around the :py:class:`~torchvision.datasets.ImageFolder` class
    that returns an :py:class:`~mmlearn.datasets.core.example.Example` object.

    Parameters
    ----------
    root_dir : str
        Path to the root directory of the dataset.
    split : {"train", "val"}, default="train"
        The split of the dataset to use.
    transform : Optional[Callable], optional, default=None
        A callable that takes in a PIL image and returns a transformed version
        of the image as a PyTorch tensor.
    target_transform : Optional[Callable], optional, default=None
        A callable that takes in the target and transforms it.
    mask_generator : Optional[Callable], optional, default=None
        A callable that generates a mask for the image.
    """

    def __init__(
        self,
        root_dir: str,
        split: Literal["train", "val"] = "train",
        transform: Optional[Callable[..., Any]] = None,
        target_transform: Optional[Callable[..., Any]] = None,
        mask_generator: Optional[Callable[..., Any]] = None,
    ) -> None:
        split = "train" if split == "train" else "val"
        root_dir = os.path.join(root_dir, split)
        super().__init__(
            root=root_dir, transform=transform, target_transform=target_transform
        )
        self.mask_generator = mask_generator

    def __getitem__(self, index: int) -> Example:
        """Get an example at the given index."""
        image, target = super().__getitem__(index)
        example = Example(
            {
                Modalities.RGB.name: image,
                Modalities.RGB.target: target,
                EXAMPLE_INDEX_KEY: index,
            }
        )
        mask = self.mask_generator() if self.mask_generator else None
        if mask is not None:  # error will be raised during collation if `None`
            example[Modalities.RGB.mask] = mask
        return example

    @property
    def zero_shot_prompt_templates(self) -> list[str]:
        """Return the zero-shot prompt templates."""
        return [
            "a bad photo of a {}.",
            "a photo of many {}.",
            "a sculpture of a {}.",
            "a photo of the hard to see {}.",
            "a low resolution photo of the {}.",
            "a rendering of a {}.",
            "graffiti of a {}.",
            "a bad photo of the {}.",
            "a cropped photo of the {}.",
            "a tattoo of a {}.",
            "the embroidered {}.",
            "a photo of a hard to see {}.",
            "a bright photo of a {}.",
            "a photo of a clean {}.",
            "a photo of a dirty {}.",
            "a dark photo of the {}.",
            "a drawing of a {}.",
            "a photo of my {}.",
            "the plastic {}.",
            "a photo of the cool {}.",
            "a close-up photo of a {}.",
            "a black and white photo of the {}.",
            "a painting of the {}.",
            "a painting of a {}.",
            "a pixelated photo of the {}.",
            "a sculpture of the {}.",
            "a bright photo of the {}.",
            "a cropped photo of a {}.",
            "a plastic {}.",
            "a photo of the dirty {}.",
            "a jpeg corrupted photo of a {}.",
            "a blurry photo of the {}.",
            "a photo of the {}.",
            "a good photo of the {}.",
            "a rendering of the {}.",
            "a {} in a video game.",
            "a photo of one {}.",
            "a doodle of a {}.",
            "a close-up photo of the {}.",
            "a photo of a {}.",
            "the origami {}.",
            "the {} in a video game.",
            "a sketch of a {}.",
            "a doodle of the {}.",
            "a origami {}.",
            "a low resolution photo of a {}.",
            "the toy {}.",
            "a rendition of the {}.",
            "a photo of the clean {}.",
            "a photo of a large {}.",
            "a rendition of a {}.",
            "a photo of a nice {}.",
            "a photo of a weird {}.",
            "a blurry photo of a {}.",
            "a cartoon {}.",
            "art of a {}.",
            "a sketch of the {}.",
            "a embroidered {}.",
            "a pixelated photo of a {}.",
            "itap of the {}.",
            "a jpeg corrupted photo of the {}.",
            "a good photo of a {}.",
            "a plushie {}.",
            "a photo of the nice {}.",
            "a photo of the small {}.",
            "a photo of the weird {}.",
            "the cartoon {}.",
            "art of the {}.",
            "a drawing of the {}.",
            "a photo of the large {}.",
            "a black and white photo of a {}.",
            "the plushie {}.",
            "a dark photo of a {}.",
            "itap of a {}.",
            "graffiti of the {}.",
            "a toy {}.",
            "itap of my {}.",
            "a photo of a cool {}.",
            "a photo of a small {}.",
            "a tattoo of the {}.",
        ]

    @property
    def id2label(self) -> dict[int, str]:
        """Return the label mapping."""
        return {
            0: "tench",
            1: "goldfish",
            2: "great white shark",
            3: "tiger shark",
            4: "hammerhead shark",
            5: "electric ray",
            6: "stingray",
            7: "rooster",
            8: "hen",
            9: "ostrich",
            10: "brambling",
            11: "goldfinch",
            12: "house finch",
            13: "junco",
            14: "indigo bunting",
            15: "American robin",
            16: "bulbul",
            17: "jay",
            18: "magpie",
            19: "chickadee",
            20: "American dipper",
            21: "kite (bird of prey)",
            22: "bald eagle",
            23: "vulture",
            24: "great grey owl",
            25: "fire salamander",
            26: "smooth newt",
            27: "newt",
            28: "spotted salamander",
            29: "axolotl",
            30: "American bullfrog",
            31: "tree frog",
            32: "tailed frog",
            33: "loggerhead sea turtle",
            34: "leatherback sea turtle",
            35: "mud turtle",
            36: "terrapin",
            37: "box turtle",
            38: "banded gecko",
            39: "green iguana",
            40: "Carolina anole",
            41: "desert grassland whiptail lizard",
            42: "agama",
            43: "frilled-necked lizard",
            44: "alligator lizard",
            45: "Gila monster",
            46: "European green lizard",
            47: "chameleon",
            48: "Komodo dragon",
            49: "Nile crocodile",
            50: "American alligator",
            51: "triceratops",
            52: "worm snake",
            53: "ring-necked snake",
            54: "eastern hog-nosed snake",
            55: "smooth green snake",
            56: "kingsnake",
            57: "garter snake",
            58: "water snake",
            59: "vine snake",
            60: "night snake",
            61: "boa constrictor",
            62: "African rock python",
            63: "Indian cobra",
            64: "green mamba",
            65: "sea snake",
            66: "Saharan horned viper",
            67: "eastern diamondback rattlesnake",
            68: "sidewinder rattlesnake",
            69: "trilobite",
            70: "harvestman",
            71: "scorpion",
            72: "yellow garden spider",
            73: "barn spider",
            74: "European garden spider",
            75: "southern black widow",
            76: "tarantula",
            77: "wolf spider",
            78: "tick",
            79: "centipede",
            80: "black grouse",
            81: "ptarmigan",
            82: "ruffed grouse",
            83: "prairie grouse",
            84: "peafowl",
            85: "quail",
            86: "partridge",
            87: "african grey parrot",
            88: "macaw",
            89: "sulphur-crested cockatoo",
            90: "lorikeet",
            91: "coucal",
            92: "bee eater",
            93: "hornbill",
            94: "hummingbird",
            95: "jacamar",
            96: "toucan",
            97: "duck",
            98: "red-breasted merganser",
            99: "goose",
            100: "black swan",
            101: "tusker",
            102: "echidna",
            103: "platypus",
            104: "wallaby",
            105: "koala",
            106: "wombat",
            107: "jellyfish",
            108: "sea anemone",
            109: "brain coral",
            110: "flatworm",
            111: "nematode",
            112: "conch",
            113: "snail",
            114: "slug",
            115: "sea slug",
            116: "chiton",
            117: "chambered nautilus",
            118: "Dungeness crab",
            119: "rock crab",
            120: "fiddler crab",
            121: "red king crab",
            122: "American lobster",
            123: "spiny lobster",
            124: "crayfish",
            125: "hermit crab",
            126: "isopod",
            127: "white stork",
            128: "black stork",
            129: "spoonbill",
            130: "flamingo",
            131: "little blue heron",
            132: "great egret",
            133: "bittern bird",
            134: "crane bird",
            135: "limpkin",
            136: "common gallinule",
            137: "American coot",
            138: "bustard",
            139: "ruddy turnstone",
            140: "dunlin",
            141: "common redshank",
            142: "dowitcher",
            143: "oystercatcher",
            144: "pelican",
            145: "king penguin",
            146: "albatross",
            147: "grey whale",
            148: "killer whale",
            149: "dugong",
            150: "sea lion",
            151: "Chihuahua",
            152: "Japanese Chin",
            153: "Maltese",
            154: "Pekingese",
            155: "Shih Tzu",
            156: "King Charles Spaniel",
            157: "Papillon",
            158: "toy terrier",
            159: "Rhodesian Ridgeback",
            160: "Afghan Hound",
            161: "Basset Hound",
            162: "Beagle",
            163: "Bloodhound",
            164: "Bluetick Coonhound",
            165: "Black and Tan Coonhound",
            166: "Treeing Walker Coonhound",
            167: "English foxhound",
            168: "Redbone Coonhound",
            169: "borzoi",
            170: "Irish Wolfhound",
            171: "Italian Greyhound",
            172: "Whippet",
            173: "Ibizan Hound",
            174: "Norwegian Elkhound",
            175: "Otterhound",
            176: "Saluki",
            177: "Scottish Deerhound",
            178: "Weimaraner",
            179: "Staffordshire Bull Terrier",
            180: "American Staffordshire Terrier",
            181: "Bedlington Terrier",
            182: "Border Terrier",
            183: "Kerry Blue Terrier",
            184: "Irish Terrier",
            185: "Norfolk Terrier",
            186: "Norwich Terrier",
            187: "Yorkshire Terrier",
            188: "Wire Fox Terrier",
            189: "Lakeland Terrier",
            190: "Sealyham Terrier",
            191: "Airedale Terrier",
            192: "Cairn Terrier",
            193: "Australian Terrier",
            194: "Dandie Dinmont Terrier",
            195: "Boston Terrier",
            196: "Miniature Schnauzer",
            197: "Giant Schnauzer",
            198: "Standard Schnauzer",
            199: "Scottish Terrier",
            200: "Tibetan Terrier",
            201: "Australian Silky Terrier",
            202: "Soft-coated Wheaten Terrier",
            203: "West Highland White Terrier",
            204: "Lhasa Apso",
            205: "Flat-Coated Retriever",
            206: "Curly-coated Retriever",
            207: "Golden Retriever",
            208: "Labrador Retriever",
            209: "Chesapeake Bay Retriever",
            210: "German Shorthaired Pointer",
            211: "Vizsla",
            212: "English Setter",
            213: "Irish Setter",
            214: "Gordon Setter",
            215: "Brittany dog",
            216: "Clumber Spaniel",
            217: "English Springer Spaniel",
            218: "Welsh Springer Spaniel",
            219: "Cocker Spaniel",
            220: "Sussex Spaniel",
            221: "Irish Water Spaniel",
            222: "Kuvasz",
            223: "Schipperke",
            224: "Groenendael dog",
            225: "Malinois",
            226: "Briard",
            227: "Australian Kelpie",
            228: "Komondor",
            229: "Old English Sheepdog",
            230: "Shetland Sheepdog",
            231: "collie",
            232: "Border Collie",
            233: "Bouvier des Flandres dog",
            234: "Rottweiler",
            235: "German Shepherd Dog",
            236: "Dobermann",
            237: "Miniature Pinscher",
            238: "Greater Swiss Mountain Dog",
            239: "Bernese Mountain Dog",
            240: "Appenzeller Sennenhund",
            241: "Entlebucher Sennenhund",
            242: "Boxer",
            243: "Bullmastiff",
            244: "Tibetan Mastiff",
            245: "French Bulldog",
            246: "Great Dane",
            247: "St. Bernard",
            248: "husky",
            249: "Alaskan Malamute",
            250: "Siberian Husky",
            251: "Dalmatian",
            252: "Affenpinscher",
            253: "Basenji",
            254: "pug",
            255: "Leonberger",
            256: "Newfoundland dog",
            257: "Great Pyrenees dog",
            258: "Samoyed",
            259: "Pomeranian",
            260: "Chow Chow",
            261: "Keeshond",
            262: "brussels griffon",
            263: "Pembroke Welsh Corgi",
            264: "Cardigan Welsh Corgi",
            265: "Toy Poodle",
            266: "Miniature Poodle",
            267: "Standard Poodle",
            268: "Mexican hairless dog (xoloitzcuintli)",
            269: "grey wolf",
            270: "Alaskan tundra wolf",
            271: "red wolf or maned wolf",
            272: "coyote",
            273: "dingo",
            274: "dhole",
            275: "African wild dog",
            276: "hyena",
            277: "red fox",
            278: "kit fox",
            279: "Arctic fox",
            280: "grey fox",
            281: "tabby cat",
            282: "tiger cat",
            283: "Persian cat",
            284: "Siamese cat",
            285: "Egyptian Mau",
            286: "cougar",
            287: "lynx",
            288: "leopard",
            289: "snow leopard",
            290: "jaguar",
            291: "lion",
            292: "tiger",
            293: "cheetah",
            294: "brown bear",
            295: "American black bear",
            296: "polar bear",
            297: "sloth bear",
            298: "mongoose",
            299: "meerkat",
            300: "tiger beetle",
            301: "ladybug",
            302: "ground beetle",
            303: "longhorn beetle",
            304: "leaf beetle",
            305: "dung beetle",
            306: "rhinoceros beetle",
            307: "weevil",
            308: "fly",
            309: "bee",
            310: "ant",
            311: "grasshopper",
            312: "cricket insect",
            313: "stick insect",
            314: "cockroach",
            315: "praying mantis",
            316: "cicada",
            317: "leafhopper",
            318: "lacewing",
            319: "dragonfly",
            320: "damselfly",
            321: "red admiral butterfly",
            322: "ringlet butterfly",
            323: "monarch butterfly",
            324: "small white butterfly",
            325: "sulphur butterfly",
            326: "gossamer-winged butterfly",
            327: "starfish",
            328: "sea urchin",
            329: "sea cucumber",
            330: "cottontail rabbit",
            331: "hare",
            332: "Angora rabbit",
            333: "hamster",
            334: "porcupine",
            335: "fox squirrel",
            336: "marmot",
            337: "beaver",
            338: "guinea pig",
            339: "common sorrel horse",
            340: "zebra",
            341: "pig",
            342: "wild boar",
            343: "warthog",
            344: "hippopotamus",
            345: "ox",
            346: "water buffalo",
            347: "bison",
            348: "ram (adult male sheep)",
            349: "bighorn sheep",
            350: "Alpine ibex",
            351: "hartebeest",
            352: "impala (antelope)",
            353: "gazelle",
            354: "arabian camel",
            355: "llama",
            356: "weasel",
            357: "mink",
            358: "European polecat",
            359: "black-footed ferret",
            360: "otter",
            361: "skunk",
            362: "badger",
            363: "armadillo",
            364: "three-toed sloth",
            365: "orangutan",
            366: "gorilla",
            367: "chimpanzee",
            368: "gibbon",
            369: "siamang",
            370: "guenon",
            371: "patas monkey",
            372: "baboon",
            373: "macaque",
            374: "langur",
            375: "black-and-white colobus",
            376: "proboscis monkey",
            377: "marmoset",
            378: "white-headed capuchin",
            379: "howler monkey",
            380: "titi monkey",
            381: "Geoffroy's spider monkey",
            382: "common squirrel monkey",
            383: "ring-tailed lemur",
            384: "indri",
            385: "Asian elephant",
            386: "African bush elephant",
            387: "red panda",
            388: "giant panda",
            389: "snoek fish",
            390: "eel",
            391: "silver salmon",
            392: "rock beauty fish",
            393: "clownfish",
            394: "sturgeon",
            395: "gar fish",
            396: "lionfish",
            397: "pufferfish",
            398: "abacus",
            399: "abaya",
            400: "academic gown",
            401: "accordion",
            402: "acoustic guitar",
            403: "aircraft carrier",
            404: "airliner",
            405: "airship",
            406: "altar",
            407: "ambulance",
            408: "amphibious vehicle",
            409: "analog clock",
            410: "apiary",
            411: "apron",
            412: "trash can",
            413: "assault rifle",
            414: "backpack",
            415: "bakery",
            416: "balance beam",
            417: "balloon",
            418: "ballpoint pen",
            419: "Band-Aid",
            420: "banjo",
            421: "baluster / handrail",
            422: "barbell",
            423: "barber chair",
            424: "barbershop",
            425: "barn",
            426: "barometer",
            427: "barrel",
            428: "wheelbarrow",
            429: "baseball",
            430: "basketball",
            431: "bassinet",
            432: "bassoon",
            433: "swimming cap",
            434: "bath towel",
            435: "bathtub",
            436: "station wagon",
            437: "lighthouse",
            438: "beaker",
            439: "military hat (bearskin or shako)",
            440: "beer bottle",
            441: "beer glass",
            442: "bell tower",
            443: "baby bib",
            444: "tandem bicycle",
            445: "bikini",
            446: "ring binder",
            447: "binoculars",
            448: "birdhouse",
            449: "boathouse",
            450: "bobsleigh",
            451: "bolo tie",
            452: "poke bonnet",
            453: "bookcase",
            454: "bookstore",
            455: "bottle cap",
            456: "hunting bow",
            457: "bow tie",
            458: "brass memorial plaque",
            459: "bra",
            460: "breakwater",
            461: "breastplate",
            462: "broom",
            463: "bucket",
            464: "buckle",
            465: "bulletproof vest",
            466: "high-speed train",
            467: "butcher shop",
            468: "taxicab",
            469: "cauldron",
            470: "candle",
            471: "cannon",
            472: "canoe",
            473: "can opener",
            474: "cardigan",
            475: "car mirror",
            476: "carousel",
            477: "tool kit",
            478: "cardboard box / carton",
            479: "car wheel",
            480: "automated teller machine",
            481: "cassette",
            482: "cassette player",
            483: "castle",
            484: "catamaran",
            485: "CD player",
            486: "cello",
            487: "mobile phone",
            488: "chain",
            489: "chain-link fence",
            490: "chain mail",
            491: "chainsaw",
            492: "storage chest",
            493: "chiffonier",
            494: "bell or wind chime",
            495: "china cabinet",
            496: "Christmas stocking",
            497: "church",
            498: "movie theater",
            499: "cleaver",
            500: "cliff dwelling",
            501: "cloak",
            502: "clogs",
            503: "cocktail shaker",
            504: "coffee mug",
            505: "coffeemaker",
            506: "spiral or coil",
            507: "combination lock",
            508: "computer keyboard",
            509: "candy store",
            510: "container ship",
            511: "convertible",
            512: "corkscrew",
            513: "cornet",
            514: "cowboy boot",
            515: "cowboy hat",
            516: "cradle",
            517: "construction crane",
            518: "crash helmet",
            519: "crate",
            520: "infant bed",
            521: "Crock Pot",
            522: "croquet ball",
            523: "crutch",
            524: "cuirass",
            525: "dam",
            526: "desk",
            527: "desktop computer",
            528: "rotary dial telephone",
            529: "diaper",
            530: "digital clock",
            531: "digital watch",
            532: "dining table",
            533: "dishcloth",
            534: "dishwasher",
            535: "disc brake",
            536: "dock",
            537: "dog sled",
            538: "dome",
            539: "doormat",
            540: "drilling rig",
            541: "drum",
            542: "drumstick",
            543: "dumbbell",
            544: "Dutch oven",
            545: "electric fan",
            546: "electric guitar",
            547: "electric locomotive",
            548: "entertainment center",
            549: "envelope",
            550: "espresso machine",
            551: "face powder",
            552: "feather boa",
            553: "filing cabinet",
            554: "fireboat",
            555: "fire truck",
            556: "fire screen",
            557: "flagpole",
            558: "flute",
            559: "folding chair",
            560: "football helmet",
            561: "forklift",
            562: "fountain",
            563: "fountain pen",
            564: "four-poster bed",
            565: "freight car",
            566: "French horn",
            567: "frying pan",
            568: "fur coat",
            569: "garbage truck",
            570: "gas mask or respirator",
            571: "gas pump",
            572: "goblet",
            573: "go-kart",
            574: "golf ball",
            575: "golf cart",
            576: "gondola",
            577: "gong",
            578: "gown",
            579: "grand piano",
            580: "greenhouse",
            581: "radiator grille",
            582: "grocery store",
            583: "guillotine",
            584: "hair clip",
            585: "hair spray",
            586: "half-track",
            587: "hammer",
            588: "hamper",
            589: "hair dryer",
            590: "hand-held computer",
            591: "handkerchief",
            592: "hard disk drive",
            593: "harmonica",
            594: "harp",
            595: "combine harvester",
            596: "hatchet",
            597: "holster",
            598: "home theater",
            599: "honeycomb",
            600: "hook",
            601: "hoop skirt",
            602: "gymnastic horizontal bar",
            603: "horse-drawn vehicle",
            604: "hourglass",
            605: "iPod",
            606: "clothes iron",
            607: "carved pumpkin",
            608: "jeans",
            609: "jeep",
            610: "T-shirt",
            611: "jigsaw puzzle",
            612: "rickshaw",
            613: "joystick",
            614: "kimono",
            615: "knee pad",
            616: "knot",
            617: "lab coat",
            618: "ladle",
            619: "lampshade",
            620: "laptop computer",
            621: "lawn mower",
            622: "lens cap",
            623: "letter opener",
            624: "library",
            625: "lifeboat",
            626: "lighter",
            627: "limousine",
            628: "ocean liner",
            629: "lipstick",
            630: "slip-on shoe",
            631: "lotion",
            632: "music speaker",
            633: "loupe magnifying glass",
            634: "sawmill",
            635: "magnetic compass",
            636: "messenger bag",
            637: "mailbox",
            638: "tights",
            639: "one-piece bathing suit",
            640: "manhole cover",
            641: "maraca",
            642: "marimba",
            643: "mask",
            644: "matchstick",
            645: "maypole",
            646: "maze",
            647: "measuring cup",
            648: "medicine cabinet",
            649: "megalith",
            650: "microphone",
            651: "microwave oven",
            652: "military uniform",
            653: "milk can",
            654: "minibus",
            655: "miniskirt",
            656: "minivan",
            657: "missile",
            658: "mitten",
            659: "mixing bowl",
            660: "mobile home",
            661: "ford model t",
            662: "modem",
            663: "monastery",
            664: "monitor",
            665: "moped",
            666: "mortar and pestle",
            667: "graduation cap",
            668: "mosque",
            669: "mosquito net",
            670: "vespa",
            671: "mountain bike",
            672: "tent",
            673: "computer mouse",
            674: "mousetrap",
            675: "moving van",
            676: "muzzle",
            677: "metal nail",
            678: "neck brace",
            679: "necklace",
            680: "baby pacifier",
            681: "notebook computer",
            682: "obelisk",
            683: "oboe",
            684: "ocarina",
            685: "odometer",
            686: "oil filter",
            687: "pipe organ",
            688: "oscilloscope",
            689: "overskirt",
            690: "bullock cart",
            691: "oxygen mask",
            692: "product packet / packaging",
            693: "paddle",
            694: "paddle wheel",
            695: "padlock",
            696: "paintbrush",
            697: "pajamas",
            698: "palace",
            699: "pan flute",
            700: "paper towel",
            701: "parachute",
            702: "parallel bars",
            703: "park bench",
            704: "parking meter",
            705: "railroad car",
            706: "patio",
            707: "payphone",
            708: "pedestal",
            709: "pencil case",
            710: "pencil sharpener",
            711: "perfume",
            712: "Petri dish",
            713: "photocopier",
            714: "plectrum",
            715: "Pickelhaube",
            716: "picket fence",
            717: "pickup truck",
            718: "pier",
            719: "piggy bank",
            720: "pill bottle",
            721: "pillow",
            722: "ping-pong ball",
            723: "pinwheel",
            724: "pirate ship",
            725: "drink pitcher",
            726: "block plane",
            727: "planetarium",
            728: "plastic bag",
            729: "plate rack",
            730: "farm plow",
            731: "plunger",
            732: "Polaroid camera",
            733: "pole",
            734: "police van",
            735: "poncho",
            736: "pool table",
            737: "soda bottle",
            738: "plant pot",
            739: "potter's wheel",
            740: "power drill",
            741: "prayer rug",
            742: "printer",
            743: "prison",
            744: "missile",
            745: "projector",
            746: "hockey puck",
            747: "punching bag",
            748: "purse",
            749: "quill",
            750: "quilt",
            751: "race car",
            752: "racket",
            753: "radiator",
            754: "radio",
            755: "radio telescope",
            756: "rain barrel",
            757: "recreational vehicle",
            758: "fishing casting reel",
            759: "reflex camera",
            760: "refrigerator",
            761: "remote control",
            762: "restaurant",
            763: "revolver",
            764: "rifle",
            765: "rocking chair",
            766: "rotisserie",
            767: "eraser",
            768: "rugby ball",
            769: "ruler measuring stick",
            770: "sneaker",
            771: "safe",
            772: "safety pin",
            773: "salt shaker",
            774: "sandal",
            775: "sarong",
            776: "saxophone",
            777: "scabbard",
            778: "weighing scale",
            779: "school bus",
            780: "schooner",
            781: "scoreboard",
            782: "CRT monitor",
            783: "screw",
            784: "screwdriver",
            785: "seat belt",
            786: "sewing machine",
            787: "shield",
            788: "shoe store",
            789: "shoji screen / room divider",
            790: "shopping basket",
            791: "shopping cart",
            792: "shovel",
            793: "shower cap",
            794: "shower curtain",
            795: "ski",
            796: "balaclava ski mask",
            797: "sleeping bag",
            798: "slide rule",
            799: "sliding door",
            800: "slot machine",
            801: "snorkel",
            802: "snowmobile",
            803: "snowplow",
            804: "soap dispenser",
            805: "soccer ball",
            806: "sock",
            807: "solar thermal collector",
            808: "sombrero",
            809: "soup bowl",
            810: "keyboard space bar",
            811: "space heater",
            812: "space shuttle",
            813: "spatula",
            814: "motorboat",
            815: "spider web",
            816: "spindle",
            817: "sports car",
            818: "spotlight",
            819: "stage",
            820: "steam locomotive",
            821: "through arch bridge",
            822: "steel drum",
            823: "stethoscope",
            824: "scarf",
            825: "stone wall",
            826: "stopwatch",
            827: "stove",
            828: "strainer",
            829: "tram",
            830: "stretcher",
            831: "couch",
            832: "stupa",
            833: "submarine",
            834: "suit",
            835: "sundial",
            836: "sunglasses",
            837: "sunglasses",
            838: "sunscreen",
            839: "suspension bridge",
            840: "mop",
            841: "sweatshirt",
            842: "swim trunks / shorts",
            843: "swing",
            844: "electrical switch",
            845: "syringe",
            846: "table lamp",
            847: "tank",
            848: "tape player",
            849: "teapot",
            850: "teddy bear",
            851: "television",
            852: "tennis ball",
            853: "thatched roof",
            854: "front curtain",
            855: "thimble",
            856: "threshing machine",
            857: "throne",
            858: "tile roof",
            859: "toaster",
            860: "tobacco shop",
            861: "toilet seat",
            862: "torch",
            863: "totem pole",
            864: "tow truck",
            865: "toy store",
            866: "tractor",
            867: "semi-trailer truck",
            868: "tray",
            869: "trench coat",
            870: "tricycle",
            871: "trimaran",
            872: "tripod",
            873: "triumphal arch",
            874: "trolleybus",
            875: "trombone",
            876: "hot tub",
            877: "turnstile",
            878: "typewriter keyboard",
            879: "umbrella",
            880: "unicycle",
            881: "upright piano",
            882: "vacuum cleaner",
            883: "vase",
            884: "vaulted or arched ceiling",
            885: "velvet fabric",
            886: "vending machine",
            887: "vestment",
            888: "viaduct",
            889: "violin",
            890: "volleyball",
            891: "waffle iron",
            892: "wall clock",
            893: "wallet",
            894: "wardrobe",
            895: "military aircraft",
            896: "sink",
            897: "washing machine",
            898: "water bottle",
            899: "water jug",
            900: "water tower",
            901: "whiskey jug",
            902: "whistle",
            903: "hair wig",
            904: "window screen",
            905: "window shade",
            906: "Windsor tie",
            907: "wine bottle",
            908: "airplane wing",
            909: "wok",
            910: "wooden spoon",
            911: "wool",
            912: "split-rail fence",
            913: "shipwreck",
            914: "sailboat",
            915: "yurt",
            916: "website",
            917: "comic book",
            918: "crossword",
            919: "traffic or street sign",
            920: "traffic light",
            921: "dust jacket",
            922: "menu",
            923: "plate",
            924: "guacamole",
            925: "consomme",
            926: "hot pot",
            927: "trifle",
            928: "ice cream",
            929: "popsicle",
            930: "baguette",
            931: "bagel",
            932: "pretzel",
            933: "cheeseburger",
            934: "hot dog",
            935: "mashed potatoes",
            936: "cabbage",
            937: "broccoli",
            938: "cauliflower",
            939: "zucchini",
            940: "spaghetti squash",
            941: "acorn squash",
            942: "butternut squash",
            943: "cucumber",
            944: "artichoke",
            945: "bell pepper",
            946: "cardoon",
            947: "mushroom",
            948: "Granny Smith apple",
            949: "strawberry",
            950: "orange",
            951: "lemon",
            952: "fig",
            953: "pineapple",
            954: "banana",
            955: "jackfruit",
            956: "cherimoya (custard apple)",
            957: "pomegranate",
            958: "hay",
            959: "carbonara",
            960: "chocolate syrup",
            961: "dough",
            962: "meatloaf",
            963: "pizza",
            964: "pot pie",
            965: "burrito",
            966: "red wine",
            967: "espresso",
            968: "tea cup",
            969: "eggnog",
            970: "mountain",
            971: "bubble",
            972: "cliff",
            973: "coral reef",
            974: "geyser",
            975: "lakeshore",
            976: "promontory",
            977: "sandbar",
            978: "beach",
            979: "valley",
            980: "volcano",
            981: "baseball player",
            982: "bridegroom",
            983: "scuba diver",
            984: "rapeseed",
            985: "daisy",
            986: "yellow lady's slipper",
            987: "corn",
            988: "acorn",
            989: "rose hip",
            990: "horse chestnut seed",
            991: "coral fungus",
            992: "agaric",
            993: "gyromitra",
            994: "stinkhorn mushroom",
            995: "earth star fungus",
            996: "hen of the woods mushroom",
            997: "bolete",
            998: "corn cob",
            999: "toilet paper",
        }
