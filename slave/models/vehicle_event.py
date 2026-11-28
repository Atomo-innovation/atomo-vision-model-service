"""The fixed vehicle message — one shape for every vehicle model.

Every vehicle published by the vehicle pipeline (processes/worker/models/
vehicle_pipeline/) carries exactly VEHICLE_DETECTION_FIELDS, whichever of
vehicle / vehicle_color / vehicle_text / number_plate the job runs: a model
that isn't in the job (or a value no model produces yet — make, speed_kmh,
lane, ...) stays null. Add a field here, and only here, when a model starts
producing it.

Message shape (one vehicle per telemetry entry, count is always 1):

    {schema_version, message_id, job_id, frame_id, occurred_at,
     source: {branch_id, device_id, zone_id,
              camera: {camera_id, status, latitude, longitude}},
     media: {object_key, url, width, height},
     telemetry: [{camera_model_assign_id, class_id, class_name, event_type,
                  tracking_id, count, detection: [{...VEHICLE_DETECTION_FIELDS}]}]}
"""

from __future__ import annotations

import uuid
from typing import Optional

from slave.models.job import JobConfig

SCHEMA_VERSION = 1
EVENT_TYPE = "vehicle_detected"

VEHICLE_DETECTION_FIELDS = (
    "confidence",            # plate box confidence (number_plate)
    "bbox",                  # plate box, frame pixels (number_plate)
    "plate_track_id",
    "plate_confirmed",
    "plate_number",
    "plate_confidence",
    "vehicle_type",
    "vehicle_type_confidence",
    "colour",
    "colour_confidence",
    "make",
    "model",
    "speed_kmh",
    "lane",
    "reg_type",
    "lp_quality",
    "lp_standard",
    "plate_signature",
    "vehicle_bbox",
    "plate_crop",
    "vehicle_crop",
    "text_detections",       # [{text_type, text_value, confidence}]
    "violations",            # [{violation_type, confidence, severity}]
)


# Master's violation_type enum. Only no_seat_belt and no_helmet are produced
# today (vehicle detector classes, see config/models/vehicle_pipeline.yaml);
# the rest need models/logic that don't exist yet (speed, signals, lanes,
# phone use, rider counting, registration lookup).
VIOLATION_TYPES = (
    "no_seat_belt",
    "over_speed",
    "red_light",
    "stop_line",
    "wrong_way",
    "no_helmet",
    "pillion_no_helmet",
    "triple_ride",
    "driver_on_call",
    "two_wheeler_driver_on_call",
    "expired_vehicle",
    "prohibited_vehicle",
)


def build_violation(violation_type: str, confidence: float, severity: Optional[str] = None) -> dict:
    if violation_type not in VIOLATION_TYPES:
        raise ValueError(f"unknown violation_type {violation_type!r} (allowed: {VIOLATION_TYPES})")
    return {"violation_type": violation_type, "confidence": confidence, "severity": severity}


def build_vehicle_detection(**filled) -> dict:
    """All VEHICLE_DETECTION_FIELDS, null unless given here."""
    unknown = set(filled) - set(VEHICLE_DETECTION_FIELDS)
    if unknown:
        raise ValueError(f"not vehicle detection fields: {sorted(unknown)}")
    detection = dict.fromkeys(VEHICLE_DETECTION_FIELDS)
    detection.update(filled)
    return detection


def build_vehicle_message(
    job: JobConfig,
    device_id: str,
    camera_status: str,
    frame_id: int,
    occurred_at: str,
    media: dict,
    camera_model_assign_id: Optional[str],
    class_id: int,
    class_name: str,
    tracking_id: str,
    detection: dict,
) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "message_id": str(uuid.uuid4()),
        "job_id": job.job_id,
        "frame_id": frame_id,
        "occurred_at": occurred_at,
        "source": {
            "branch_id": job.branch_id,
            "device_id": device_id,
            "zone_id": job.zone_id,
            "camera": {
                "camera_id": job.camera_id,
                "status": camera_status,
                "latitude": job.latitude,
                "longitude": job.longitude,
            },
        },
        "media": media,
        "telemetry": [
            {
                "camera_model_assign_id": camera_model_assign_id,
                "class_id": class_id,
                "class_name": class_name,
                "event_type": EVENT_TYPE,
                "tracking_id": tracking_id,
                "count": 1,
                "detection": [detection],
            }
        ],
    }
