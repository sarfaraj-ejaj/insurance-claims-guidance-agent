"""Step 0: frozen target schema and controlled vocabularies (ADR v1.0 sections 2-3)."""
from types import MappingProxyType as _ro

VERSION = "v1.0"

VOCAB = _ro({
    "channel": ("Email", "Phone", "Portal", "In-Person"),
    "coverage_type": ("Property", "Auto", "Liability"),
    "object_type": ("Building", "Vehicle"),
    "incident_type": (
        "vandalism", "theft_burglary", "kitchen_fire", "electrical_fire",
        "burst_pipe_flood", "roof_leak", "storm_damage", "slip_and_fall",
        "property_injury", "product_liability", "rear_end_collision",
        "side_impact_collision", "head_on_collision", "parking_lot_collision"),
    "location_type": (
        "residence_interior", "residence_exterior", "commercial_property",
        "parking_lot", "residential_street", "intersection", "highway"),
})

# field -> type description, exactly as in ADR section 2
SCHEMA = _ro({
    "header": _ro({
        "claim_id": "string | null", "report_date": "YYYY-MM-DD",
        "incident_date": "YYYY-MM-DD | null", "reported_by": "string", "channel": "enum"}),
    "policy_details": _ro({
        "policy_number": "string", "policyholder_name": "string", "coverage_type": "enum",
        "effective_date": "YYYY-MM-DD | null", "expiration_date": "YYYY-MM-DD | null"}),
    "insured_objects": _ro({
        "object_id": "string | null", "object_type": "enum", "make_model": "string | null",
        "year": "integer | null", "location_address": "string | null",
        "estimated_value": "integer | null"}),
    "incident_description": _ro({
        "incident_type": "enum", "location_type": "enum",
        "estimated_damage_amount": "integer | null", "police_report_number": "string | null"}),
})


def check_enum(field, value):
    """Return value if it is in the field's vocabulary, else raise ValueError (never invent a label)."""
    if field not in VOCAB:
        raise ValueError(f"unknown enum field: {field!r}")
    if value not in VOCAB[field]:
        raise ValueError(f"{value!r} is not a valid {field}; log it in ADR section 9")
    return value
