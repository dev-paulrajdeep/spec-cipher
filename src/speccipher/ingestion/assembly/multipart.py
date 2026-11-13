"""Multipart publication assembly from declared page ranges."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from speccipher.domain.errors import MultipartAssemblyError


def _page_number(volume: Mapping[str, Any], field: str, *, label: str) -> int:
    value = volume.get(field)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise MultipartAssemblyError(f"{label} {field} must be a positive integer")
    return value


def _prepare_volume(volume: Mapping[str, Any], index: int) -> dict[str, Any]:
    if not isinstance(volume, Mapping):
        raise MultipartAssemblyError(f"Volume {index} must be an object")
    start = _page_number(volume, "page_start", label=f"Volume {index}")
    end = _page_number(volume, "page_end", label=f"Volume {index}")
    if end < start:
        raise MultipartAssemblyError(f"Volume {index} page_end cannot precede page_start")

    range_count = end - start + 1
    page_count = volume.get("page_count", range_count)
    if not isinstance(page_count, int) or isinstance(page_count, bool) or page_count < 1:
        raise MultipartAssemblyError(f"Volume {index} page_count must be a positive integer")
    if page_count != range_count:
        raise MultipartAssemblyError(
            f"Volume {index} page_count {page_count} does not match its declared range "
            f"({start}-{end}, {range_count} pages)"
        )

    prepared = dict(volume)
    prepared.update(
        {
            "page_start": start,
            "page_end": end,
            "page_count": page_count,
            "physical_page_start": 1,
            "physical_page_end": page_count,
            "absolute_page_start": start,
            "absolute_page_end": end,
            "page_offset": start - 1,
        }
    )
    return prepared


def _prepare_missing_volume(volume: Mapping[str, Any], index: int) -> dict[str, Any]:
    if not isinstance(volume, Mapping):
        raise MultipartAssemblyError(f"Missing volume declaration {index} must be an object")
    volume_id = volume.get("volume_id")
    if not isinstance(volume_id, str) or not volume_id.strip():
        raise MultipartAssemblyError(
            f"Missing volume declaration {index} requires a non-empty volume_id"
        )
    start = _page_number(volume, "page_start", label=f"Missing volume {volume_id!r}")
    end = _page_number(volume, "page_end", label=f"Missing volume {volume_id!r}")
    if end < start:
        raise MultipartAssemblyError(
            f"Missing volume {volume_id!r} page_end cannot precede page_start"
        )

    return {
        "volume_id": volume_id.strip(),
        "page_start": start,
        "page_end": end,
        "page_count": end - start + 1,
        "reason": volume.get("reason"),
        "missing": True,
    }


def assemble_multipart_publication(
    volumes: Sequence[Mapping[str, Any]],
    *,
    missing_volumes: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Order source volumes by declared absolute page ranges and map provenance.

    A gap is accepted only when a missing volume with an explicit identity and
    page range accounts for it. Missing spans are reported as metadata; no page
    or volume is synthesized. Filenames are preserved as provenance and never
    participate in ordering.
    """
    if not volumes:
        raise MultipartAssemblyError("Empty volume list cannot be assembled")

    actual = [_prepare_volume(volume, index) for index, volume in enumerate(volumes, start=1)]
    missing = [
        _prepare_missing_volume(volume, index)
        for index, volume in enumerate(missing_volumes, start=1)
    ]
    spans = [(item["page_start"], item["page_end"], item, False) for item in actual] + [
        (item["page_start"], item["page_end"], item, True) for item in missing
    ]
    spans.sort(key=lambda item: (item[0], item[1], bool(item[3])))

    if spans[0][0] != 1:
        raise MultipartAssemblyError(
            f"Publication page ranges must begin at page 1; first declared page is {spans[0][0]}"
        )

    for previous, current in zip(spans, spans[1:], strict=False):
        previous_start, previous_end = previous[0], previous[1]
        current_start, current_end = current[0], current[1]
        if current_start <= previous_end:
            raise MultipartAssemblyError(
                f"Overlapping declared page ranges {previous_start}-{previous_end} and "
                f"{current_start}-{current_end}"
            )
        expected_next = previous_end + 1
        if current_start != expected_next:
            raise MultipartAssemblyError(
                f"Non-contiguous page range between {previous_start}-{previous_end} and "
                f"{current_start}-{current_end}: gap of {current_start - expected_next} pages; "
                "declare the missing volume and its page range explicitly"
            )

    asserted_identities = [
        item["asserted_identity"] for item in actual if item.get("asserted_identity") is not None
    ]
    if any(
        not isinstance(identity, str) or not identity.strip() for identity in asserted_identities
    ):
        raise MultipartAssemblyError("asserted_identity values must be non-empty strings")
    identities = set(asserted_identities)
    if len(identities) > 1:
        raise MultipartAssemblyError(
            f"Conflicting publication identities across volumes: {sorted(identities)}"
        )

    order_by_identity = {
        id(item): order for order, (_start, _end, item, _is_missing) in enumerate(spans, start=1)
    }
    ordered_actual = sorted(actual, key=lambda item: item["page_start"])
    for volume in ordered_actual:
        volume["volume_order"] = order_by_identity[id(volume)]

    ordered_missing = sorted(missing, key=lambda item: item["page_start"])
    for volume in ordered_missing:
        volume["volume_order"] = order_by_identity[id(volume)]

    last_page = spans[-1][1]
    total_ingested_pages = sum(volume["page_count"] for volume in actual)
    return {
        "status": "partial" if ordered_missing else "assembled",
        "volumes_count": len(ordered_actual),
        "total_pages": total_ingested_pages,
        "declared_page_span": last_page,
        "volumes": ordered_actual,
        "missing_volumes": ordered_missing,
    }
