"""Discover satellite imagery from the configured STAC providers."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from enum import StrEnum
import math
import re
from typing import Any, Mapping

from pystac_client import Client
from requests.exceptions import Timeout
from shapely.geometry import shape


DEFAULT_MAX_ITEMS_PER_PROVIDER = 100
DEFAULT_REQUEST_TIMEOUT_SECONDS = 15.0


@dataclass(frozen=True)
class ProviderConfig:
    """STAC endpoint and provider-specific collection identifiers."""

    name: str
    endpoint: str
    collections: Mapping[str, str]


PROVIDERS: tuple[ProviderConfig, ...] = (
    ProviderConfig(
        name="microsoft_planetary_computer",
        endpoint="https://planetarycomputer.microsoft.com/api/stac/v1",
        collections={
            "sentinel_2_l2a": "sentinel-2-l2a",
            "sentinel_1_sar": "sentinel-1-rtc",
            "landsat_c2_l2": "landsat-c2-l2",
        },
    ),
    ProviderConfig(
        name="element_84_earth_search",
        endpoint="https://earth-search.aws.element84.com/v1",
        collections={
            "sentinel_2_l2a": "sentinel-2-l2a",
            "sentinel_1_sar": "sentinel-1-grd",
            "landsat_c2_l2": "landsat-c2-l2",
        },
    ),
)


class InputValidationError(ValueError):
    """Raised when discovery input cannot form a valid STAC search."""


class DiscoveryStatus(StrEnum):
    """Outcome of one provider search."""

    SUCCESS = "success"
    EMPTY = "empty"
    TIMEOUT = "timeout"
    ERROR = "error"


@dataclass(frozen=True)
class ProviderDiscoveryResult:
    """Independent discovery outcome for one STAC provider."""

    provider: str
    endpoint: str
    collections: Mapping[str, str]
    status: DiscoveryStatus
    items: tuple[dict[str, Any], ...]
    error: str | None = None


_DATE_ONLY = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def discover_imagery(
    geometry: Mapping[str, Any],
    start_date: str | date | datetime,
    end_date: str | date | datetime,
    *,
    max_items_per_provider: int = DEFAULT_MAX_ITEMS_PER_PROVIDER,
    request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
) -> dict[str, ProviderDiscoveryResult]:
    """Search all providers concurrently and return their outcomes separately.

    Date-only bounds cover the full UTC day. Datetime bounds must include a UTC
    offset. The item dictionaries returned by each provider are left intact.
    """

    validated_geometry = _validate_geometry(geometry)
    datetime_interval = _validate_date_range(start_date, end_date)
    _validate_positive_integer(max_items_per_provider, "max_items_per_provider")
    _validate_positive_number(request_timeout_seconds, "request_timeout_seconds")

    results: dict[str, ProviderDiscoveryResult] = {}
    with ThreadPoolExecutor(max_workers=len(PROVIDERS)) as executor:
        futures = {
            executor.submit(
                _search_provider,
                provider,
                validated_geometry,
                datetime_interval,
                max_items_per_provider,
                request_timeout_seconds,
            ): provider
            for provider in PROVIDERS
        }
        for future in as_completed(futures):
            provider = futures[future]
            try:
                result = future.result()
            except Timeout as exc:
                result = _failed_result(
                    provider, DiscoveryStatus.TIMEOUT, _error_message(exc)
                )
            except Exception as exc:  # A provider failure must not erase peer results.
                result = _failed_result(
                    provider, DiscoveryStatus.ERROR, _error_message(exc)
                )
            results[provider.name] = result

    return {provider.name: results[provider.name] for provider in PROVIDERS}


def _search_provider(
    provider: ProviderConfig,
    geometry: dict[str, Any],
    datetime_interval: str,
    max_items: int,
    request_timeout: float,
) -> ProviderDiscoveryResult:
    client = Client.open(
        provider.endpoint,
        timeout=(request_timeout, request_timeout),
    )
    search = client.search(
        method="POST",
        collections=list(provider.collections.values()),
        intersects=geometry,
        datetime=datetime_interval,
        max_items=max_items,
    )
    items = tuple(search.items_as_dicts())
    status = DiscoveryStatus.SUCCESS if items else DiscoveryStatus.EMPTY
    return ProviderDiscoveryResult(
        provider=provider.name,
        endpoint=provider.endpoint,
        collections=dict(provider.collections),
        status=status,
        items=items,
    )


def _failed_result(
    provider: ProviderConfig, status: DiscoveryStatus, error: str
) -> ProviderDiscoveryResult:
    return ProviderDiscoveryResult(
        provider=provider.name,
        endpoint=provider.endpoint,
        collections=dict(provider.collections),
        status=status,
        items=(),
        error=error,
    )


def _validate_geometry(geometry: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(geometry, Mapping):
        raise InputValidationError("geometry must be a GeoJSON geometry object")

    geometry_type = geometry.get("type")
    if geometry_type not in {"Point", "Polygon"}:
        raise InputValidationError("geometry type must be Point or Polygon")
    if "coordinates" not in geometry:
        raise InputValidationError("geometry must contain coordinates")

    coordinates = geometry["coordinates"]
    if geometry_type == "Point":
        _validate_position(coordinates)
    else:
        _validate_polygon_coordinates(coordinates)

    normalized = {"type": geometry_type, "coordinates": coordinates}
    try:
        parsed = shape(normalized)
    except (TypeError, ValueError, AttributeError) as exc:
        raise InputValidationError(f"invalid GeoJSON geometry: {exc}") from exc
    if parsed.is_empty:
        raise InputValidationError("geometry must not be empty")
    if not parsed.is_valid:
        raise InputValidationError("geometry is topologically invalid")
    return normalized


def _validate_polygon_coordinates(coordinates: Any) -> None:
    if not isinstance(coordinates, (list, tuple)) or not coordinates:
        raise InputValidationError("Polygon coordinates must contain at least one ring")
    for ring in coordinates:
        if not isinstance(ring, (list, tuple)) or len(ring) < 4:
            raise InputValidationError("each Polygon ring must contain at least 4 positions")
        for position in ring:
            _validate_position(position)
        if ring[0] != ring[-1]:
            raise InputValidationError("each Polygon ring must be closed")


def _validate_position(position: Any) -> None:
    if not isinstance(position, (list, tuple)) or len(position) != 2:
        raise InputValidationError("coordinates must be longitude/latitude pairs")
    longitude, latitude = position
    if (
        isinstance(longitude, bool)
        or isinstance(latitude, bool)
        or not isinstance(longitude, (int, float))
        or not isinstance(latitude, (int, float))
        or not math.isfinite(longitude)
        or not math.isfinite(latitude)
    ):
        raise InputValidationError("coordinates must contain finite numbers")
    if not -180 <= longitude <= 180:
        raise InputValidationError("longitude must be between -180 and 180")
    if not -90 <= latitude <= 90:
        raise InputValidationError("latitude must be between -90 and 90")


def _validate_date_range(
    start_value: str | date | datetime, end_value: str | date | datetime
) -> str:
    start = _parse_temporal_bound(start_value, is_end=False)
    end = _parse_temporal_bound(end_value, is_end=True)
    if start > end:
        raise InputValidationError("start_date must be before or equal to end_date")
    return f"{_format_utc(start)}/{_format_utc(end)}"


def _parse_temporal_bound(
    value: str | date | datetime, *, is_end: bool
) -> datetime:
    if isinstance(value, str):
        try:
            if _DATE_ONLY.fullmatch(value):
                parsed_date = date.fromisoformat(value)
                value = parsed_date
            else:
                value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise InputValidationError(
                "dates must be ISO dates or RFC 3339 datetimes"
            ) from exc

    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise InputValidationError("datetime values must include a UTC offset")
        return value.astimezone(timezone.utc)
    if isinstance(value, date):
        boundary = time.max if is_end else time.min
        return datetime.combine(value, boundary, tzinfo=timezone.utc)
    raise InputValidationError("dates must be strings, dates, or datetimes")


def _format_utc(value: datetime) -> str:
    return value.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _validate_positive_integer(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise InputValidationError(f"{name} must be a positive integer")


def _validate_positive_number(value: Any, name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise InputValidationError(f"{name} must be a positive finite number")


def _error_message(error: Exception) -> str:
    message = str(error).strip()
    return message or error.__class__.__name__
