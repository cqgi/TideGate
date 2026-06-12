from __future__ import annotations


def rpm_key(tenant: str) -> str:
    return f"quota:{tenant}:rpm"


def tpm_key(tenant: str) -> str:
    return f"quota:{tenant}:tpm"


def conc_key(tenant: str) -> str:
    return f"quota:{tenant}:conc"


def budget_key(tenant: str, month: str) -> str:
    return f"quota:{tenant}:budget:{month}"


def reservation_zset_key(tenant: str) -> str:
    return f"resv:{tenant}"


def reservation_data_key(tenant: str) -> str:
    return f"resv:data:{tenant}"


def correction_key(tenant: str, model: str) -> str:
    return f"corr:{tenant}:{model}"
