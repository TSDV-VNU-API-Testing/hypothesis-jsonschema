from __future__ import annotations

import base64
import logging
import os
import random
import sys
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from importlib import import_module
from time import time as current_time
from typing import (
    Any,
    Callable,
    Collection,
    Iterable,
    Literal,
    Mapping,
    Sequence,
    Union,
    get_args,
)
from uuid import UUID

from faker import Faker
from hypothesis import strategies as st

VAS_KEY_PREFIX = "ZTc3N2RlYmUtMWJmMC00NjNjLTkzYjYtOWNmN2IxOGQ0ODkzCg=="


def get_key_with_vas_prefix(key_name: str):
    return f"{VAS_KEY_PREFIX}_{key_name}"


DEV = True

CURRENT_LEVEL = logging.DEBUG if DEV else logging.INFO
CURRENT_FORMAT = (
    "%(asctime)s %(filename)s:%(lineno)d:%(funcName)s %(levelname)s:%(message)s"
)
CURRENT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


logging.basicConfig(
    level=CURRENT_LEVEL,
    format=CURRENT_FORMAT,
    datefmt=CURRENT_DATE_FORMAT,
    stream=sys.stdout,
)
logger = logging.getLogger(name="Logger")
logger.setLevel(level=CURRENT_LEVEL)


# Full option
# {
#   codec=None,
#   min_codepoint=None,
#   max_codepoint=None,
#   categories=None,
#   exclude_characters=None,
#   include_characters=None,
# }

ASCII_OPTION = {
    "min_codepoint": ord("\u0020"),  # 0 ~ Space
    "max_codepoint": ord("\u007E"),  # ~ ~ Tilde
}
ASCII_NO_SYMBOL_AND_PUNCTUATION_OPTION = {
    # For codepoint visit:
    # https://hypothesis.readthedocs.io/en/latest/data.html#hypothesis.strategies.characters
    # https://en.wikipedia.org/wiki/List_of_Unicode_characters
    # To get code point of a unicode char, use `ord()`
    # Want to char in ascii range, but include letter and number, exclude symbols like (<,>,=, ...)
    "min_codepoint": ord("\u0030"),  # 0 ~ Digit Zero
    "max_codepoint": ord("\u007A"),  # z ~ Latin Small Letter Z
    "exclude_characters": [
        "\u003A",
        "\u003B",
        "\u003C",
        "\u003D",
        "\u003E",
        "\u003F",
        "\u0040",
        "\u005B",
        "\u005C",
        "\u005D",
        "\u005E",
        "\u005F",
        "\u0060",
        "\u007B",
        "\u007C",
        "\u007D",
        "\u007E",
    ],
    # For categories visit: https://en.wikipedia.org/wiki/Unicode_character_property
    # categories=("Lu", "Ll", "Lt", "Lm", "Nd", "Nl"),
}

VasCodec = Literal["ascii", "ascii_no_symbol_and_punctuation"]
VAS_CODEC: list[VasCodec] = list(get_args(VasCodec))
CODEC_OPTION_MAP: dict[VasCodec, dict[str, Any]] = {
    "ascii": ASCII_OPTION,
    "ascii_no_symbol_and_punctuation": ASCII_NO_SYMBOL_AND_PUNCTUATION_OPTION,
}

# List only used locales
FAKER_ALL_BUILTIN_LOCALES = [
    "en",
    "en_US",
    "en_CA",
    "en_GB",
    "en_IE",
    "en_NZ",
    "en_PH",
    "en_TH",
    "en_US",
    "vi_VN",
]
FAKER_BUILTIN_PROVIDERS = [
    "address",
    "automotive",
    "bank",
    "barcode",
    "color",
    "company",
    "credit_card",
    "currency",
    "date_time",
    "emoji",
    # "file",
    "geo",
    "internet",
    "isbn",
    "job",
    "lorem",
    "misc",
    "passport",
    "person",
    "phone_number",
    "profile",
    "python",
    "sbn",
    "ssn",
    "user_agent",
]
FAKER_BUILTIN_PROVIDER_METHODS = [
    # Base provider
    "bothify",
    "hexify",
    "language_code",
    "lexify",
    "locale",
    "numerify",
    "random_choices",
    "random_digit",
    "random_digit_above_two",
    "random_digit_not_null",
    "random_digit_not_null_or_empty",
    "random_digit_or_empty",
    "random_element",
    "random_elements",
    "random_int",
    "random_letter",
    "random_letters",
    "random_lowercase_letter",
    "random_number",
    "random_sample",
    "random_uppercase_letter",
    "randomize_nb_elements",
    # -------------
    # address
    "address",
    "building_number",
    "city",
    "city_suffix",
    "country",
    "country_code",
    "current_country",
    "current_country_code",
    "postcode",
    "street_address",
    "street_name",
    "street_suffix",
    # -------------
    # automotive
    "license_plate",
    "vin",
    # -------------
    # bank
    "aba",
    "bank_country",
    "bban",
    "iban",
    "swift",
    "swift11",
    "swift8",
    # -------------
    # barcode
    "ean",
    "ean13",
    "ean8",
    "localized_ean",
    "localized_ean13",
    "localized_ean8",
    # -------------
    # color
    "color",
    "color_name",
    "hex_color",
    "rgb_color",
    "rgb_css_color",
    "safe_color_name",
    "safe_hex_color",
    # -------------
    # company
    "bs",
    "catch_phrase",
    "company",
    "company_suffix",
    # -------------
    # credit_card
    "credit_card_expire",
    "credit_card_full",
    "credit_card_number",
    "credit_card_provider",
    "credit_card_security_code",
    # -------------
    # currency
    "cryptocurrency",
    "cryptocurrency_code",
    "cryptocurrency_name",
    "currency",
    "currency_code",
    "currency_name",
    "currency_symbol",
    "pricetag",
    # -------------
    # date_time
    "am_pm",
    "century",
    "date",
    "date_between",
    "date_between_dates",
    "date_object",
    "date_of_birth",
    "date_this_century",
    "date_this_decade",
    "date_this_month",
    "date_this_year",
    "date_time",
    "date_time_ad",
    "date_time_between",
    "date_time_between_dates",
    "date_time_this_century",
    "date_time_this_decade",
    "date_time_this_month",
    "date_time_this_year",
    "day_of_month",
    "day_of_week",
    "future_date",
    "future_datetime",
    "iso8601",
    "month",
    "month_name",
    "past_date",
    "past_datetime",
    "pytimezone",
    "time",
    "time_delta",
    "time_object",
    "time_series",
    "timezone",
    "unix_time",
    "year",
    # -------------
    # emoji
    "emoji",
    # -------------
    # file
    "file_extension",
    "file_name",
    "file_path",
    "mime_type",
    "unix_device",
    "unix_partition",
    # -------------
    # geo
    "coordinate",
    "latitude",
    "latlng",
    "local_latlng",
    "location_on_land",
    "longitude",
    # -------------
    # internet
    "ascii_company_email",
    "ascii_email",
    "ascii_free_email",
    "ascii_safe_email",
    "company_email",
    "dga",
    "domain_name",
    "domain_word",
    "email",
    "free_email",
    "free_email_domain",
    "hostname",
    "http_method",
    "iana_id",
    # "image_url",
    "ipv4",
    "ipv4_network_class",
    "ipv4_private",
    "ipv4_public",
    "ipv6",
    "mac_address",
    "nic_handle",
    "nic_handles",
    "port_number",
    "ripe_id",
    "safe_domain_name",
    "safe_email",
    "slug",
    "tld",
    "uri",
    "uri_extension",
    "uri_page",
    "uri_path",
    "url",
    "user_name",
    # -------------
    # isbn
    "isbn10",
    "isbn13",
    # -------------
    # job
    "job",
    # -------------
    # lorem
    "paragraph",
    "paragraphs",
    "sentence",
    "sentences",
    "text",
    "texts",
    "word",
    "words",
    # -------------
    # misc
    # "binary",
    # "boolean",
    "boolean",
    "csv",
    "dsv",
    "fixed_width",
    # "image",
    "json",
    "json_bytes",
    "md5",
    "null_boolean",
    "password",
    "psv",
    "sha1",
    "sha256",
    "tar",
    "tsv",
    "uuid4",
    "xml",
    "zip",
    # -------------
    # passport
    "passport_dob",
    "passport_number",
    "passport_owner",
    # -------------
    # person
    "first_name",
    "first_name_female",
    "first_name_male",
    "first_name_nonbinary",
    "language_name",
    "last_name",
    "last_name_female",
    "last_name_male",
    "last_name_nonbinary",
    "name",
    "name_female",
    "name_male",
    "name_nonbinary",
    "prefix",
    "prefix_female",
    "prefix_male",
    "prefix_nonbinary",
    "suffix",
    "suffix_female",
    "suffix_male",
    "suffix_nonbinary",
    # -------------
    # phone_number
    "country_calling_code",
    "msisdn",
    "phone_number",
    # -------------
    # profile
    "profile",
    "simple_profile",
    # -------------
    # python
    # "enum",
    "pybool",
    "pydecimal",
    "pydict",
    "pyfloat",
    "pyint",
    "pyiterable",
    "pylist",
    "pyobject",
    "pyset",
    "pystr",
    "pystr_format",
    "pystruct",
    "pytuple",
    # -------------
    # sbn
    "sbn9",
    # -------------
    # ssn
    "ssn",
    # -------------
    # user_agent
    "android_platform_token",
    "chrome",
    "firefox",
    "internet_explorer",
    "ios_platform_token",
    "linux_platform_token",
    "linux_processor",
    "mac_platform_token",
    "mac_processor",
    "opera",
    "safari",
    "user_agent",
    "windows_platform_token",
    # -------------
]
FAKER_COMMUNITY_PROVIDERS = [
    "faker_airtravel.AirTravelProvider",
    "faker_biology.physiology.CellType",
    "faker_biology.physiology.Organ",
    "faker_biology.physiology.Organelle",
    "faker_biology.bioseq.Bioseq",
    "faker_biology.mol_biol.Antibody",
    "faker_biology.mol_biol.RestrictionEnzyme",
    "faker_biology.mol_biol.Enzyme",
    "faker_biology.taxonomy.ModelOrganism",
    "faker_credit_score.CreditScore",
    # "faker_education.SchoolProvider",
    "faker_marketdata.MarketDataProvider",
    "faker_microservice.Provider",
    "faker_music.MusicProvider",
    "mdgen.MarkdownPostProvider",
    "faker_vehicle.VehicleProvider",
    "faker_web.WebProvider",
    # "faker_wifi_essid.WifiESSID",
]
FAKER_COMMUNITY_PROVIDER_METHODS = [
    # air travel
    "airport_object",
    "airport_name",
    "airport_iata",
    "airport_icao",
    "airline",
    "flight",
    # biology
    "organ",
    "celltype",
    "common_eukaryotic_organelle",
    "plant_organelle",
    "animal_organelle",
    "organelle",
    "dna",
    "rna",
    "stop_codon",
    "cds",
    "protein",
    "protein_name",
    "protein_desc",
    "protein_name_desc",
    "amino_acid",
    "amino_acid_name",
    "amino_acid_3_letters",
    "amino_acid_1_letter",
    "amino_acid_mass",
    "antibody_isotype",
    "antibody_application",
    "antibody_source",
    "dilution",
    "restriction_enzyme",
    "restriction_enzyme_data",
    "blunt",
    "sticky",
    "enzyme_category",
    "enzyme",
    "organism_english",
    "organism_latin",
    "organism",
    # credit score
    "credit_score_name",
    "credit_score_provider",
    "credit_score",
    "credit_score_full",
    # # education
    # "school_object",
    # "school_name",
    # "school_nces_id",
    # "school_district",
    # "school_level",
    # "school_type",
    # "school_state",
    # market data
    "isin",
    "sedol",
    "mic",
    "lei",
    "cusip",
    "ric",
    "ticker",
    "nsin",
    "figi",
    "marketType",
    # microservice
    "microservice",
    # music
    "music_genre_object",
    "music_genre",
    "music_subgenre",
    "music_instrument_object",
    "music_instrument",
    "music_instrument_category",
    "post",
    # vehicle
    "vehicle_object",
    "vehicle_year_make_model",
    "vehicle_year_make_model_cat",
    "vehicle_make_model",
    "vehicle_make",
    "vehicle_year",
    "vehicle_model",
    "vehicle_category",
    "machine_object",
    "machine_year_make_model",
    "machine_year_make_model_cat",
    "machine_make_model",
    "machine_make",
    "machine_year",
    "machine_model",
    "machine_category",
    # web
    "content_type",
    "content_type_popular",
    "apache",
    "nginx",
    "iis",
    "server_token",
    # # wifi essid
    # "common_essid",
    # "upc_default_essid",
    # "bbox_default_essid",
    # "wifi_essid",
]
FAKER_ALL_PROVIDER_METHODS = [
    *FAKER_BUILTIN_PROVIDER_METHODS,
    *FAKER_COMMUNITY_PROVIDER_METHODS,
]

FAKER = Faker(
    use_weighting=False,
)
# Import all provider - use list to import instead of manual way
for _, provider_name in enumerate(FAKER_BUILTIN_PROVIDERS):
    provider_module = import_module(f"faker.providers.{provider_name}")
    FAKER.add_provider(provider_module)
# With community provider, have to import class
for _, provider_name in enumerate(FAKER_COMMUNITY_PROVIDERS):
    splits = provider_name.split(".")
    module_path = ".".join([submodule for submodule in splits[0 : len(splits) - 1]])
    class_name = splits[-1]
    provider_module = import_module(module_path)
    provider_class = getattr(provider_module, class_name)
    FAKER.add_provider(provider_class)
# Check all method is callable
for _, method_name in enumerate(FAKER_ALL_PROVIDER_METHODS):
    method = FAKER.__getattr__(method_name)
    assert isinstance(method, Callable)


def get_faker_strategy(key: str) -> st.SearchStrategy[Union[str, None]]:

    matched_method = None

    for _, method_name in enumerate(FAKER_ALL_PROVIDER_METHODS):
        if is_key_match_method_name(key, method_name):
            matched_method = method_name
            break

    if not matched_method:
        return st.none()

    def serialize(data: Any) -> str:
        # Có thể gen ra
        # int, float, bool, str, bytes, UUID: OK
        # datetime.date, datetime.datetime, datetime.timedelta, datetime.timezone, datetime.time: OK
        # list, tuple, set, Iterable, Sequence, Collection: OK
        # dict: OK
        # decimal.Decimal: OK,
        # TODO: datetime.tzinfo, Iterator, Enum

        if data == None:
            return ""
        if isinstance(data, str):
            return data

        if isinstance(data, (int, float, bool, timedelta, timezone, UUID)):
            return str(data)
        if isinstance(data, bytes):
            return base64.b64encode(data).decode("utf-8")
        if isinstance(data, (date, datetime, time)):
            return data.isoformat()
        if isinstance(data, Decimal):
            return str(float(data))
        if isinstance(data, (list, tuple, set, Iterable, Sequence, Collection)):
            return ", ".join([serialize(ele) for _, ele in enumerate(data)])
        if isinstance(data, (list, tuple, set)):
            return ", ".join([serialize(ele) for _, ele in enumerate(data)])
        if isinstance(data, (dict, Mapping)):
            return ", ".join(
                [f"{serialize(k)}: {serialize(v)}" for k, v in data.items()]
            )

        return data

    method = FAKER.__getattr__(matched_method)
    assert isinstance(method, Callable)
    return st.builds(lambda: serialize(method()))
    # return st.just(serialize(method()))


def is_key_match_method_name(key: str, method_name: str) -> bool:
    # INFO: method_name sẽ có snake_case
    # key API của 1 object có thể là các dạng: snake_case, camelCase, kebab-case, PascalCase
    # Cần xử lý: ta sẽ đưa method name snake_case về camelCase, kebab-case, và PascalCase

    def get_variant_keys():
        def get_normalized_keys(unnormalized_key: str):
            normalized_key = unnormalized_key.strip().replace(" ", "")
            lower_key = unnormalized_key.lower()

            return [normalized_key, lower_key]

        def get_no_underscore_keys(undescore_keys: list[str]):
            return map(
                lambda undescore_key: undescore_key.replace("_", ""), undescore_keys
            )

        def get_no_hyphen_keys(hyphen_keys: list[str]):
            return map(lambda hyphen_key: hyphen_key.replace("-", ""), hyphen_keys)

        normalized_keys = get_normalized_keys(key)
        return [
            *normalized_keys,
            *get_no_underscore_keys(normalized_keys),
            *get_no_hyphen_keys(normalized_keys),
        ]

    def get_variant_method_names():
        def snake_to_camel():
            splits = method_name.lower().split("_")
            return splits[0] + "".join(ele.title() for ele in splits[1:])

        def snake_to_kebab():
            splits = method_name.lower().split("_")
            return "-".join(splits)

        def snake_to_pascal():
            splits = method_name.lower().split("_")
            return "".join(ele.title() for ele in splits)

        def get_no_underscore_method_names(undescore_method_names: list[str]):
            return map(
                lambda undescore_method_name: undescore_method_name.replace("_", ""),
                undescore_method_names,
            )

        def get_no_hyphen_method_names(hyphen_method_names: list[str]):
            return map(
                lambda hyphen_method_name: hyphen_method_name.replace("-", ""),
                hyphen_method_names,
            )

        normalized_method_names = [
            method_name,
            snake_to_camel(),
            snake_to_pascal(),
            snake_to_kebab(),
        ]
        return [
            *get_no_underscore_method_names(normalized_method_names),
            *get_no_hyphen_method_names(normalized_method_names),
        ]

    def is_key_match(key: str):
        return key in get_variant_method_names()

    return any(
        map(
            lambda k: is_key_match(k),
            get_variant_keys(),
        )
    )


VAS_IMAGE_FOLDER = "./public/img"
VAS_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
VAS_IMAGE_PATHS = [
    os.path.join(VAS_IMAGE_FOLDER, file)
    for file in os.listdir(VAS_IMAGE_FOLDER)
    if os.path.isfile(os.path.join(VAS_IMAGE_FOLDER, file))
    if file.lower().endswith(VAS_IMAGE_EXTENSIONS)
]

#Image type standard mapping
IMAGE_TYPE_STANDARD: dict[str, str] = {
    "ico": "vnd.microsoft.icon",
    "jpg": "jpeg",
    "svg": "svg+xml",
    "tif": "tiff",
}


class VasImage:
    # The purpose of "_: int" is for st.builds()
    def __init__(self, _: int):
        self.image_path = self._get_image_path()
        self.image_name = self._get_image_name()
        self.image_type = self._get_image_type()
        self.image_size = self._get_image_size()
        self.image_binary = self._get_image_binary()

    def _get_image_path(self) -> str:
        random.seed(current_time())
        ran_num = random.randint(0, min(len(VAS_IMAGE_PATHS) - 1, 0))
        return VAS_IMAGE_PATHS[ran_num]

    def _get_image_name(self) -> str:
        logger.debug("image name is called")
        return os.path.basename(self.image_path)

    def _get_image_type(self) -> str:
        logger.debug("image type is called")
        components = self.image_name.split(".")
        content_type = components[len(components) - 1]

        if content_type in IMAGE_TYPE_STANDARD:
            return f"image/{IMAGE_TYPE_STANDARD[content_type]}"

        return f"image/{content_type}"

    def _get_image_size(self) -> str:
        logger.debug("image size is called")
        file_size = os.path.getsize(self.image_path) / (1024 * 1024)  # Convert to MB
        formatted_file_size = (
            f"{file_size:.2f} MB"  # Format to 2 decimal places and add MB unit
        )
        return formatted_file_size

    def _get_image_binary(self) -> str | bytes:
        logger.debug("image binary is called")
        with open(self.image_path, "rb") as image_file:
            return "IMAGE".encode("utf-8")
            # return "IMAGE".encode("utf-8") if DEV else image_file.read()

    def get_image_object(self) -> object:
        return {
            "image_name": self.image_name,
            "image_type": self.image_type,
            "image_size": self.image_size,
            "image_url": self.image_path,
        }
