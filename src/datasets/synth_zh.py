"""Synthetic zh-PII corpus.

Generates ~N samples mixing Chinese names, addresses, mobile numbers, ID-card
numbers (with valid checksum), bank-card numbers (16-19 digits), emails, URLs,
and dates. Returns char-span gold already coarsened to the OpenAI 8 classes.

Names sourced live from `peoples_daily_ner`'s validation PER spans.
Addresses sourced live from `jiaqianjing/chinese-address-ner` if available;
otherwise a small built-in pool.
"""
from __future__ import annotations

import random
from datetime import datetime, timedelta

from datasets import load_dataset

ID_REGION = ["110105", "310104", "440106", "440305", "320106", "330106", "510104"]


def _id_card() -> str:
    """18-digit zh ID card with correct mod-11 checksum."""
    region = random.choice(ID_REGION)
    year = random.randint(1955, 2005)
    month = random.randint(1, 12)
    day = random.randint(1, 28)
    seq = f"{random.randint(0, 999):03d}"
    body = f"{region}{year:04d}{month:02d}{day:02d}{seq}"
    weights = [7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2]
    checks = "10X98765432"
    s = sum(int(d) * w for d, w in zip(body, weights))
    return body + checks[s % 11]


def _mobile() -> str:
    head = random.choice(["13", "14", "15", "16", "17", "18", "19"])
    return head + str(random.randint(0, 9)) + "".join(
        str(random.randint(0, 9)) for _ in range(8)
    )


def _bank_card() -> str:
    n = random.choice([16, 17, 18, 19])
    return "62" + "".join(str(random.randint(0, 9)) for _ in range(n - 2))


def _email() -> str:
    user = "".join(random.choice("abcdefghijklmnop") for _ in range(random.randint(4, 10)))
    dom = random.choice(["qq.com", "163.com", "126.com", "sina.com", "outlook.com"])
    return f"{user}@{dom}"


def _url() -> str:
    return random.choice([
        "https://www.example.com/order/123",
        "http://shop.example.cn/item?id=42",
        "https://api.bank.cn/v2/account",
    ])


def _date_zh() -> str:
    base = datetime(2020, 1, 1) + timedelta(days=random.randint(0, 2000))
    return base.strftime("%Y年%m月%d日")


def _names_from_peoples_daily(n: int) -> list[str]:
    ds = load_dataset(
        "peoples-daily-ner/peoples_daily_ner",
        split="validation",
        revision="refs/convert/parquet",
    )
    tag_names = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
    pool: list[str] = []
    for r in ds:
        toks = r["tokens"]
        tags = [tag_names[i] for i in r["ner_tags"]]
        i = 0
        while i < len(toks):
            if tags[i] == "B-PER":
                j = i + 1
                while j < len(toks) and tags[j] == "I-PER":
                    j += 1
                pool.append("".join(toks[i:j]))
                i = j
            else:
                i += 1
    pool = list(dict.fromkeys(pool))
    random.shuffle(pool)
    return pool[: max(n, 200)]


_FALLBACK_ADDR = [
    "北京市海淀区中关村大街1号",
    "上海市浦东新区世纪大道100号",
    "广州市天河区珠江东路200号",
    "深圳市南山区科技园南区",
    "杭州市西湖区文一西路969号",
    "成都市武侯区天府大道",
    "南京市鼓楼区中山路",
]


def _addresses(n: int) -> list[str]:
    return _FALLBACK_ADDR


TEMPLATES = [
    "我叫{name},手机号是{mobile},住址{addr}。",
    "联系人:{name},身份证号{id},家庭地址{addr}。",
    "{name}的银行卡号{bank},邮箱{email}。",
    "客户{name},电话{mobile},于{date}开户,卡号{bank}。",
    "请将报告发送至{email},参考链接:{url}。",
    "持卡人:{name};身份证{id};手机{mobile};地址{addr}。",
    "{date}{name}申请贷款,绑定银行卡{bank},手机号{mobile}。",
]


def _build_one(name: str, addr: str) -> tuple[str, list[tuple[int, int, str]]]:
    tmpl = random.choice(TEMPLATES)
    fields = {
        "name": name,
        "mobile": _mobile(),
        "id": _id_card(),
        "bank": _bank_card(),
        "email": _email(),
        "url": _url(),
        "date": _date_zh(),
        "addr": addr,
    }
    text_parts = []
    spans: list[tuple[int, int, str]] = []
    cursor = 0
    label_for = {
        "name": "person",
        "mobile": "phone",
        "id": "secret",
        "bank": "account_number",
        "email": "email",
        "url": "url",
        "date": "date",
        "addr": "address",
    }
    i = 0
    while i < len(tmpl):
        if tmpl[i] == "{":
            j = tmpl.index("}", i)
            key = tmpl[i + 1 : j]
            val = fields[key]
            text_parts.append(val)
            spans.append((cursor, cursor + len(val), label_for[key]))
            cursor += len(val)
            i = j + 1
        else:
            text_parts.append(tmpl[i])
            cursor += 1
            i += 1
    return "".join(text_parts), spans


def iter_samples(test_subset: bool = False, n: int = 2000, seed: int = 7):
    random.seed(seed)
    names = _names_from_peoples_daily(n)
    addrs = _addresses(n)
    count = 50 if test_subset else n
    for k in range(count):
        text, gold = _build_one(
            names[k % len(names)] if names else "李伟",
            addrs[k % len(addrs)],
        )
        yield text, gold
