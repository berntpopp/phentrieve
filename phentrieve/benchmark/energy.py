"""Optional benchmark energy tracking helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class BaseEnergyTracker:
    measurement_source: str

    def start_run(self) -> None:
        return None

    def stop_run(self) -> dict[str, Any]:
        return {"measurement_source": self.measurement_source}


class DisabledEnergyTracker(BaseEnergyTracker):
    def __init__(self) -> None:
        super().__init__(measurement_source="disabled")


class UnavailableEnergyTracker(BaseEnergyTracker):
    def __init__(self, *, reason: str) -> None:
        super().__init__(measurement_source="unavailable")
        self.reason = reason

    def stop_run(self) -> dict[str, Any]:
        payload = super().stop_run()
        payload["reason"] = self.reason
        return payload


class CodeCarbonEnergyTracker(BaseEnergyTracker):
    def __init__(self, *, config: Any, tracker_cls: Any) -> None:
        super().__init__(measurement_source="measured")
        kwargs: dict[str, Any] = {}
        if getattr(config, "country_iso_code", None):
            kwargs["country_iso_code"] = config.country_iso_code
        if getattr(config, "region", None):
            kwargs["region"] = config.region
        self._tracker = tracker_cls(**kwargs)

    def start_run(self) -> None:
        self._tracker.start()

    def stop_run(self) -> dict[str, Any]:
        emissions_kg = self._tracker.stop()
        return {
            "measurement_source": self.measurement_source,
            "energy_kwh": None,
            "carbon_kg": emissions_kg,
        }


def create_energy_tracker(config: Any) -> BaseEnergyTracker:
    if not getattr(config, "measure_energy", False):
        return DisabledEnergyTracker()
    try:
        from codecarbon import OfflineEmissionsTracker
    except ImportError:
        return UnavailableEnergyTracker(reason="codecarbon_not_installed")
    return CodeCarbonEnergyTracker(config=config, tracker_cls=OfflineEmissionsTracker)
