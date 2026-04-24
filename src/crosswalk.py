"""Coarsen every system's labels to OpenAI privacy-filter's 8 categories.

OPENAI canonical taxonomy (8): person, address, email, phone, url, date,
account_number, secret. Plus `O` (no PII).

A label not in the taxonomy is mapped to None and dropped from scoring.
"""
from __future__ import annotations

OPENAI_LABELS = (
    "person",
    "address",
    "email",
    "phone",
    "url",
    "date",
    "account_number",
    "secret",
)


def _norm(s: str) -> str:
    return s.lower().lstrip("bi-").lstrip("bies-").replace("private_", "")


# OpenAI native: private_person, private_address, private_email, private_phone,
# private_url, private_date, account_number, secret.
OPENAI_MAP = {
    "private_person": "person",
    "private_address": "address",
    "private_email": "email",
    "private_phone": "phone",
    "private_url": "url",
    "private_date": "date",
    "account_number": "account_number",
    "secret": "secret",
}

# Presidio (~50 entity types) → OpenAI 8.
PRESIDIO_MAP = {
    "PERSON": "person",
    "EMAIL_ADDRESS": "email",
    "PHONE_NUMBER": "phone",
    "URL": "url",
    "LOCATION": "address",
    "DATE_TIME": "date",
    "CREDIT_CARD": "account_number",
    "IBAN_CODE": "account_number",
    "US_BANK_NUMBER": "account_number",
    "US_SSN": "secret",
    "US_PASSPORT": "secret",
    "US_DRIVER_LICENSE": "secret",
    "UK_NHS": "secret",
    "UK_NINO": "secret",
    "MEDICAL_LICENSE": "secret",
    "CRYPTO": "account_number",
    "IP_ADDRESS": "url",
    # Custom zh recognizers we register.
    "CN_MOBILE": "phone",
    "CN_ID_CARD": "secret",
    "CN_BANK_CARD": "account_number",
}

# AI4Privacy (21 BIO labels, 19 entities). Keys are entity types after stripping B-/I-.
AI4PRIVACY_MAP = {
    "GIVENNAME": "person",
    "SURNAME": "person",
    "TITLE": "person",
    "EMAIL": "email",
    "TELEPHONENUM": "phone",
    "STREET": "address",
    "CITY": "address",
    "BUILDINGNUM": "address",
    "ZIPCODE": "address",
    "DATE": "date",
    "AGE": None,
    "GENDER": None,
    "SEX": None,
    "IDCARDNUM": "secret",
    "PASSPORTNUM": "secret",
    "DRIVERLICENSENUM": "secret",
    "SOCIALNUM": "secret",
    "TAXNUM": "secret",
    "CREDITCARDNUMBER": "account_number",
}

# bert4ner-base-chinese: PER / LOC / ORG / TIME (coarse).
BERT4NER_MAP = {
    "PER": "person",
    "LOC": "address",
    "TIME": "date",
    "ORG": None,
}

# Gretel synthetic_pii_finance_multilingual entity labels → OpenAI 8.
GRETEL_MAP = {
    "name": "person",
    "first_name": "person",
    "last_name": "person",
    "email": "email",
    "phone_number": "phone",
    "url": "url",
    "address": "address",
    "street_address": "address",
    "city": "address",
    "state": "address",
    "country": "address",
    "zipcode": "address",
    "date": "date",
    "date_of_birth": "date",
    "credit_card_number": "account_number",
    "iban": "account_number",
    "swift_bic_code": "account_number",
    "bban": "account_number",
    "routing_number": "account_number",
    "account_number": "account_number",
    "passport": "secret",
    "ssn": "secret",
    "driver_license_number": "secret",
    "api_key": "secret",
    "password": "secret",
    "company": None,
    "job_title": None,
}

# AI4Privacy 400k privacy_mask labels (mostly same as the 21-label set above).
AI4PRIVACY_400K_MAP = {**AI4PRIVACY_MAP, "USERNAME": None, "IP": "url"}

# peoples_daily_ner (CoNLL): PER / LOC / ORG.
PEOPLES_DAILY_MAP = {"PER": "person", "LOC": "address", "ORG": None}


def coarsen(label: str, mapping: dict) -> str | None:
    if label is None:
        return None
    raw = label
    # Strip BIO/BIOES prefixes if present.
    if "-" in raw and raw.split("-", 1)[0].upper() in ("B", "I", "E", "S"):
        raw = raw.split("-", 1)[1]
    return mapping.get(raw)
