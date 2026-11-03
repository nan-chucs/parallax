from __future__ import annotations

from datetime import datetime, timezone
from threading import Barrier
from typing import Any

import pytest
from requests.exceptions import ReadTimeout

from app.discovery import (
    DiscoveryStatus,
    InputValidationError,
    PROVIDERS,
    discover_imagery,
)


POINT = {"type": "Point", "coordinates": [77.5946, 12.9716]}
POLYGON = {
    "type": "Polygon",
    "coordinates": [
        [
            [77.5, 12.9],
            [77.7, 12.9],
            [77.7, 13.1],
            [77.5, 13.1],
            [77.5, 12.9],
        ]
    ],
}


class FakeSearch:
    def __init__(self, items: list[dict[str, Any]]) -> None:
        self._items = items

    def items_as_dicts(self):
        yield from self._items


class FakeClient:
    def __init__(self, items: list[dict[str, Any]], calls: list[dict[str, Any]]) -> None:
        self._items = items
        self._calls = calls

    def search(self, **kwargs: Any) -> FakeSearch:
        self._calls.append(kwargs)
        return FakeSearch(self._items)


def test_successful_discovery_preserves_metadata_and_provider_collections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, list[dict[str, Any]]] = {}
    items = {
        PROVIDERS[0].endpoint: [
            {
                "type": "Feature",
                "id": "pc-item",
                "collection": "sentinel-1-rtc",
                "properties": {"datetime": "2024-01-02T00:00:00Z", "sat:orbit_state": "ascending"},
                "assets": {"vv": {"href": "https://example.test/pc.tif"}},
            }
        ],
        PROVIDERS[1].endpoint: [
            {
                "type": "Feature",
                "id": "e84-item",
                "collection": "sentinel-1-grd",
                "properties": {"datetime": "2024-01-02T00:00:00Z"},
                "assets": {"vv": {"href": "s3://example/e84.tif"}},
            }
        ],
    }

    def open_client(endpoint: str, **kwargs: Any) -> FakeClient:
        assert kwargs["timeout"] == (15.0, 15.0)
        calls[endpoint] = []
        return FakeClient(items[endpoint], calls[endpoint])

    monkeypatch.setattr("app.discovery.Client.open", open_client)

    results = discover_imagery(POLYGON, "2024-01-01", "2024-01-31")

    assert list(results) == [provider.name for provider in PROVIDERS]
    for provider in PROVIDERS:
        result = results[provider.name]
        assert result.status is DiscoveryStatus.SUCCESS
        assert result.provider == provider.name
        assert result.collections == provider.collections
        assert result.items[0] is items[provider.endpoint][0]
        assert calls[provider.endpoint] == [
            {
                "method": "POST",
                "collections": list(provider.collections.values()),
                "intersects": POLYGON,
                "datetime": "2024-01-01T00:00:00.000000Z/2024-01-31T23:59:59.999999Z",
                "max_items": 100,
            }
        ]


def test_searches_providers_concurrently(monkeypatch: pytest.MonkeyPatch) -> None:
    barrier = Barrier(2, timeout=1)

    class ConcurrentClient(FakeClient):
        def search(self, **kwargs: Any) -> FakeSearch:
            barrier.wait()
            return FakeSearch([])

    monkeypatch.setattr(
        "app.discovery.Client.open",
        lambda endpoint, **kwargs: ConcurrentClient([], []),
    )

    results = discover_imagery(POINT, "2024-01-01", "2024-01-02")

    assert all(result.status is DiscoveryStatus.EMPTY for result in results.values())


def test_empty_results_are_distinct_from_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.discovery.Client.open",
        lambda endpoint, **kwargs: FakeClient([], []),
    )

    results = discover_imagery(POINT, "2024-01-01", "2024-01-01")

    assert all(result.status is DiscoveryStatus.EMPTY for result in results.values())
    assert all(result.error is None for result in results.values())


def test_timeout_does_not_discard_other_provider_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    successful_item = {"id": "available", "collection": "landsat-c2-l2"}

    def open_client(endpoint: str, **kwargs: Any) -> FakeClient:
        if endpoint == PROVIDERS[0].endpoint:
            raise ReadTimeout("provider timed out")
        return FakeClient([successful_item], [])

    monkeypatch.setattr("app.discovery.Client.open", open_client)

    results = discover_imagery(POINT, "2024-01-01", "2024-01-02")

    assert results[PROVIDERS[0].name].status is DiscoveryStatus.TIMEOUT
    assert results[PROVIDERS[0].name].items == ()
    assert results[PROVIDERS[0].name].error == "provider timed out"
    assert results[PROVIDERS[1].name].status is DiscoveryStatus.SUCCESS
    assert results[PROVIDERS[1].name].items == (successful_item,)


def test_provider_error_does_not_discard_empty_peer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def open_client(endpoint: str, **kwargs: Any) -> FakeClient:
        if endpoint == PROVIDERS[1].endpoint:
            raise RuntimeError("provider rejected request")
        return FakeClient([], [])

    monkeypatch.setattr("app.discovery.Client.open", open_client)

    results = discover_imagery(POINT, "2024-01-01", "2024-01-02")

    assert results[PROVIDERS[0].name].status is DiscoveryStatus.EMPTY
    assert results[PROVIDERS[1].name].status is DiscoveryStatus.ERROR
    assert results[PROVIDERS[1].name].error == "provider rejected request"


@pytest.mark.parametrize(
    "geometry",
    [
        {"type": "LineString", "coordinates": [[0, 0], [1, 1]]},
        {"type": "Point", "coordinates": [181, 0]},
        {"type": "Point", "coordinates": [0, 91]},
        {"type": "Point", "coordinates": [0, 0, 1]},
        {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1]]]},
        {
            "type": "Polygon",
            "coordinates": [[[0, 0], [1, 1], [1, 0], [0, 1], [0, 0]]],
        },
    ],
)
def test_invalid_geometry_is_rejected_before_network_access(
    monkeypatch: pytest.MonkeyPatch, geometry: dict[str, Any]
) -> None:
    def unexpected_open(*args: Any, **kwargs: Any) -> None:
        pytest.fail("network client should not be opened")

    monkeypatch.setattr("app.discovery.Client.open", unexpected_open)

    with pytest.raises(InputValidationError):
        discover_imagery(geometry, "2024-01-01", "2024-01-02")


@pytest.mark.parametrize(
    ("start", "end", "message"),
    [
        ("not-a-date", "2024-01-02", "ISO dates"),
        ("2024-02-30", "2024-03-01", "ISO dates"),
        ("2024-02-02", "2024-02-01", "before or equal"),
        (
            datetime(2024, 1, 1),
            datetime(2024, 1, 2, tzinfo=timezone.utc),
            "UTC offset",
        ),
    ],
)
def test_invalid_date_range_is_rejected(start: Any, end: Any, message: str) -> None:
    with pytest.raises(InputValidationError, match=message):
        discover_imagery(POINT, start, end)


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_invalid_max_items_is_rejected(value: Any) -> None:
    with pytest.raises(InputValidationError, match="positive integer"):
        discover_imagery(POINT, "2024-01-01", "2024-01-02", max_items_per_provider=value)
