from fastapi import FastAPI
from fastapi.responses import JSONResponse
import hashlib
import json
import math
import re
from datetime import datetime, timezone


app = FastAPI()

# Persistent state while the service process is running.
runs = {}


MAX_SAFE_INTEGER = 9007199254740991

TIME_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T"
    r"\d{2}:\d{2}:\d{2}"
    r"(?:\.\d{1,3})?"
    r"(?:Z|[+-]\d{2}:\d{2})$"
)


def utf8_key(x):
    return x.encode("utf-8")


def valid_safe_int(x):
    return (
        isinstance(x, int)
        and not isinstance(x, bool)
        and 0 <= x <= MAX_SAFE_INTEGER
    )


def parse_time(value):
    if not isinstance(value, str):
        return None

    if not TIME_RE.fullmatch(value):
        return None

    s = value

    if s.endswith("Z"):
        s = s[:-1] + "+00:00"

    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None

    if dt.tzinfo is None:
        return None

    offset = dt.utcoffset()

    if offset is None:
        return None

    seconds = abs(offset.total_seconds())

    if seconds > 14 * 60 * 60:
        return None

    if seconds == 14 * 60 * 60 and seconds % 3600 != 0:
        return None

    return dt.astimezone(timezone.utc)


def canonical_time(value):
    dt = parse_time(value)

    if dt is None:
        return None

    return (
        dt.strftime("%Y-%m-%dT%H:%M:%S.")
        + f"{dt.microsecond // 1000:03d}Z"
    )


def finite_number(x):
    return (
        isinstance(x, (int, float))
        and not isinstance(x, bool)
        and math.isfinite(float(x))
    )


def compact_json(obj):
    return json.dumps(
        obj,
        ensure_ascii=False,
        separators=(",", ":")
    )


def digest_dataset(train_ids, eval_ids, feature_names):
    obj = {
        "trainRowIds": train_ids,
        "evalRowIds": eval_ids,
        "featureNames": feature_names,
    }

    return hashlib.sha256(
        compact_json(obj).encode("utf-8")
    ).hexdigest()


def invalid_input():
    return JSONResponse(
        status_code=400,
        content={"error": "INVALID_INPUT"}
    )


# ============================================================
# SELECT
# ============================================================

def select_phase(data):

    errors = []

    run_id = data.get("runId")

    if (
        not isinstance(run_id, str)
        or not run_id
        or len(run_id) > 128
    ):
        errors.append("INVALID_INPUT")

    forbidden = data.get("forbiddenFeatures")

    if not isinstance(forbidden, list) or any(
        not isinstance(x, str) for x in forbidden
    ):
        errors.append("INVALID_INPUT")

    limit = data.get("numTrialsLimit")

    if not valid_safe_int(limit) or limit <= 0:
        errors.append("INVALID_INPUT")

    rows = data.get("rows")

    if not isinstance(rows, list) or len(rows) == 0:
        errors.append("INVALID_INPUT")

    trials = data.get("trials")

    if not isinstance(trials, list):
        errors.append("INVALID_INPUT")

    if errors:
        return {
            "runId": run_id,
            "selectedTrialId": None,
            "trainRowIds": [],
            "evalRowIds": [],
            "featureNames": [],
            "datasetDigest": None,
            "reasonCodes": ["INVALID_INPUT"],
        }

    # --------------------------------------------------------
    # Validate rows
    # --------------------------------------------------------

    retained = {}
    row_valid = True
    row_ids = set()

    for row in rows:

        if not isinstance(row, dict):
            row_valid = False
            break

        required = {
            "id",
            "entity",
            "eventTime",
            "predictionTime",
            "version",
            "split",
            "features",
        }

        if set(row.keys()) != required:
            row_valid = False
            break

        rid = row["id"]

        if not isinstance(rid, str) or rid in row_ids:
            row_valid = False
            break

        row_ids.add(rid)

        if not isinstance(row["entity"], str):
            row_valid = False
            break

        if canonical_time(row["eventTime"]) is None:
            row_valid = False
            break

        if canonical_time(row["predictionTime"]) is None:
            row_valid = False
            break

        if not valid_safe_int(row["version"]):
            row_valid = False
            break

        if row["split"] not in ("TRAIN", "EVAL"):
            row_valid = False
            break

        if not isinstance(row["features"], dict):
            row_valid = False
            break

        for fname, feature in row["features"].items():

            if not isinstance(fname, str):
                row_valid = False
                break

            if not isinstance(feature, dict):
                row_valid = False
                break

            if set(feature.keys()) != {"value", "availableAt"}:
                row_valid = False
                break

            if canonical_time(feature["availableAt"]) is None:
                row_valid = False
                break

        if not row_valid:
            break

        entity = row["entity"]
        event_time = canonical_time(row["eventTime"])

        key = (entity, event_time)

        candidate = {
            "id": rid,
            "entity": entity,
            "eventTime": event_time,
            "version": row["version"],
            "predictionTime": canonical_time(
                row["predictionTime"]
            ),
            "split": row["split"],
            "features": row["features"],
        }

        if key not in retained:
            retained[key] = candidate
        else:
            old = retained[key]

            if candidate["version"] > old["version"]:
                retained[key] = candidate

            elif (
                candidate["version"] == old["version"]
                and utf8_key(candidate["id"])
                < utf8_key(old["id"])
            ):
                retained[key] = candidate

    if not row_valid:
        return {
            "runId": run_id,
            "selectedTrialId": None,
            "trainRowIds": [],
            "evalRowIds": [],
            "featureNames": [],
            "datasetDigest": None,
            "reasonCodes": ["INVALID_INPUT"],
        }

    retained_rows = list(retained.values())

    # --------------------------------------------------------
    # Features eligible in EVERY retained row
    # --------------------------------------------------------

    feature_sets = [
        set(row["features"].keys())
        for row in retained_rows
    ]

    common_features = set.intersection(*feature_sets)

    forbidden_set = set(forbidden)

    eligible = []

    for fname in common_features:

        if fname in forbidden_set:
            continue

        ok = True

        for row in retained_rows:

            available = canonical_time(
                row["features"][fname]["availableAt"]
            )

            if available is None:
                ok = False
                break

            prediction = parse_time(
                row["predictionTime"]
            )

            available_dt = parse_time(
                row["features"][fname]["availableAt"]
            )

            if available_dt > prediction:
                ok = False
                break

        if ok:
            eligible.append(fname)

    feature_names = sorted(
        eligible,
        key=utf8_key
    )

    # --------------------------------------------------------
    # Train/EVAL IDs
    # --------------------------------------------------------

    train_ids = sorted(
        [
            row["id"]
            for row in retained_rows
            if row["split"] == "TRAIN"
        ],
        key=utf8_key
    )

    eval_ids = sorted(
        [
            row["id"]
            for row in retained_rows
            if row["split"] == "EVAL"
        ],
        key=utf8_key
    )

    # --------------------------------------------------------
    # Trials
    # --------------------------------------------------------

    trial_ids = set()

    for trial in trials:

        if not isinstance(trial, dict):
            return {
                "runId": run_id,
                "selectedTrialId": None,
                "trainRowIds": [],
                "evalRowIds": [],
                "featureNames": [],
                "datasetDigest": None,
                "reasonCodes": ["INVALID_INPUT"],
            }

        if set(trial.keys()) != {
            "trialId",
            "status",
            "evalMetric"
        }:
            return {
                "runId": run_id,
                "selectedTrialId": None,
                "trainRowIds": [],
                "evalRowIds": [],
                "featureNames": [],
                "datasetDigest": None,
                "reasonCodes": ["INVALID_INPUT"],
            }

        tid = trial["trialId"]

        if not valid_safe_int(tid) or tid in trial_ids:
            return {
                "runId": run_id,
                "selectedTrialId": None,
                "trainRowIds": [],
                "evalRowIds": [],
                "featureNames": [],
                "datasetDigest": None,
                "reasonCodes": ["INVALID_INPUT"],
            }

        trial_ids.add(tid)

        if trial["status"] not in ("SUCCEEDED", "FAILED"):
            return {
                "runId": run_id,
                "selectedTrialId": None,
                "trainRowIds": [],
                "evalRowIds": [],
                "featureNames": [],
                "datasetDigest": None,
                "reasonCodes": ["INVALID_INPUT"],
            }

        if not finite_number(trial["evalMetric"]):
            # Non-finite trial isn't eligible, but the input itself
            # is not malformed.
            pass

    reason_codes = []

    if len(trials) > limit:
        reason_codes.append("TRIAL_LIMIT_EXCEEDED")

    successful = [
        t for t in trials
        if (
            t["status"] == "SUCCEEDED"
            and finite_number(t["evalMetric"])
        )
    ]

    if not successful:
        reason_codes.append("NO_SUCCESSFUL_TRIAL")

    if reason_codes:
        reason_codes = sorted(
            set(reason_codes),
            key=utf8_key
        )

        return {
            "runId": run_id,
            "selectedTrialId": None,
            "trainRowIds": train_ids,
            "evalRowIds": eval_ids,
            "featureNames": feature_names,
            "datasetDigest": None,
            "reasonCodes": reason_codes,
        }

    selected = max(
        successful,
        key=lambda t: (
            float(t["evalMetric"]),
            -t["trialId"]
        )
    )

    dataset_digest = digest_dataset(
        train_ids,
        eval_ids,
        feature_names
    )

    response = {
        "runId": run_id,
        "selectedTrialId": selected["trialId"],
        "trainRowIds": train_ids,
        "evalRowIds": eval_ids,
        "featureNames": feature_names,
        "datasetDigest": dataset_digest,
        "reasonCodes": [],
    }

    return response


# ============================================================
# EVALUATE
# ============================================================

def evaluate_phase(data):

    reason_codes = []

    run_id = data.get("runId")

    if not isinstance(run_id, str) or not run_id:
        reason_codes.append("INVALID_INPUT")

    selected_trial = data.get("selectedTrialId")

    if not valid_safe_int(selected_trial):
        reason_codes.append("INVALID_INPUT")

    digest = data.get("datasetDigest")

    if (
        not isinstance(digest, str)
        or not re.fullmatch(r"[0-9a-f]{64}", digest)
    ):
        reason_codes.append("INVALID_INPUT")

    metric_floor = data.get("metricFloor")

    if not finite_number(metric_floor) or not (
        0 <= float(metric_floor) <= 1
    ):
        reason_codes.append("INVALID_INPUT")

    required_slices = data.get("requiredSlices")

    if not isinstance(required_slices, dict):
        reason_codes.append("INVALID_INPUT")
    else:
        for name, floor in required_slices.items():
            if (
                not isinstance(name, str)
                or not name
                or not finite_number(floor)
                or not 0 <= float(floor) <= 1
            ):
                reason_codes.append("INVALID_INPUT")

    rows = data.get("rows")

    if not isinstance(rows, list):
        reason_codes.append("INVALID_INPUT")

    bytes_processed = data.get("bytesProcessed")
    max_bytes = data.get("maxBytes")

    if not valid_safe_int(bytes_processed):
        reason_codes.append("INVALID_INPUT")

    if not valid_safe_int(max_bytes):
        reason_codes.append("INVALID_INPUT")

    # --------------------------------------------------------
    # Defaults
    # --------------------------------------------------------

    test_metric = None
    critical_slice_pass = False

    # --------------------------------------------------------
    # Stored selection lineage
    # --------------------------------------------------------

    stored = runs.get(run_id)

    if stored is None:
        reason_codes.append("INVALID_LINEAGE")
    else:
        if (
            selected_trial != stored["selectedTrialId"]
            or digest != stored["datasetDigest"]
        ):
            reason_codes.append("INVALID_LINEAGE")

    # --------------------------------------------------------
    # Test rows
    # --------------------------------------------------------

    valid_test_rows = True

    if isinstance(rows, list):

        if len(rows) == 0:
            valid_test_rows = False

        for row in rows:

            if not isinstance(row, dict):
                valid_test_rows = False
                break

            if set(row.keys()) != {
                "label",
                "prediction",
                "slice"
            }:
                valid_test_rows = False
                break

            if (
                not isinstance(row["label"], int)
                or isinstance(row["label"], bool)
                or row["label"] not in (0, 1)
            ):
                valid_test_rows = False
                break

            if (
                not isinstance(row["prediction"], int)
                or isinstance(row["prediction"], bool)
                or row["prediction"] not in (0, 1)
            ):
                valid_test_rows = False
                break

            if (
                not isinstance(row["slice"], str)
                or not row["slice"]
            ):
                valid_test_rows = False
                break

    else:
        valid_test_rows = False

    if not valid_test_rows:
        reason_codes.append("INVALID_TEST_ROW")

    # --------------------------------------------------------
    # Metric calculations
    # --------------------------------------------------------

    if valid_test_rows:

        correct = sum(
            row["label"] == row["prediction"]
            for row in rows
        )

        test_metric = round(
            correct / len(rows),
            12
        )

        # Required slices
        critical_slice_pass = True

        for name, floor in required_slices.items():

            slice_rows = [
                row
                for row in rows
                if row["slice"] == name
            ]

            if not slice_rows:
                reason_codes.append(
                    f"MISSING_SLICE:{name}"
                )
                critical_slice_pass = False
                continue

            slice_accuracy = round(
                sum(
                    row["label"] == row["prediction"]
                    for row in slice_rows
                ) / len(slice_rows),
                12
            )

            if slice_accuracy < float(floor):
                reason_codes.append(
                    f"SLICE_FLOOR:{name}"
                )
                critical_slice_pass = False

        if test_metric < float(metric_floor):
            reason_codes.append("AGGREGATE_FLOOR")

        if not any(
            x.startswith("MISSING_SLICE:")
            or x.startswith("SLICE_FLOOR:")
            for x in reason_codes
        ):
            critical_slice_pass = True

    # --------------------------------------------------------
    # Byte limit
    # --------------------------------------------------------

    if (
        valid_safe_int(bytes_processed)
        and valid_safe_int(max_bytes)
        and bytes_processed > max_bytes
    ):
        reason_codes.append("BYTE_LIMIT")

    # --------------------------------------------------------
    # Final decision
    # --------------------------------------------------------

    reason_codes = sorted(
        set(reason_codes),
        key=utf8_key
    )

    decision = (
        "admit"
        if not reason_codes
        else "reject"
    )

    return {
        "runId": run_id,
        "selectedTrialId": (
            selected_trial
            if valid_safe_int(selected_trial)
            else None
        ),
        "datasetDigest": digest,
        "testMetric": test_metric,
        "criticalSlicePass": critical_slice_pass,
        "decision": decision,
        "bytesProcessed": bytes_processed,
        "reasonCodes": reason_codes,
    }


# ============================================================
# MAIN ENDPOINT
# ============================================================

@app.post("/bqml")
async def bqml(data: dict):

    if not isinstance(data, dict):
        return invalid_input()

    phase = data.get("phase")

    if phase not in ("select", "evaluate"):
        return invalid_input()

    if phase == "select":

        result = select_phase(data)

        run_id = result["runId"]

        # Store only valid run IDs.
        if (
            isinstance(run_id, str)
            and run_id
            and len(run_id) <= 128
        ):

            if run_id in runs:

                # Identical replay
                if runs[run_id]["select_response"] == result:
                    return runs[run_id]["select_response"]

                return JSONResponse(
                    status_code=409,
                    content={"error": "RUN_ID_CONFLICT"}
                )

            # Persist complete selection response
            runs[run_id] = {
                "select_response": result,
                "selectedTrialId": result["selectedTrialId"],
                "datasetDigest": result["datasetDigest"],
            }

        return result

    return evaluate_phase(data)