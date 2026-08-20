from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import copy
import hashlib
import json
import math
import re
from datetime import datetime, timezone


app = FastAPI()

MAX_SAFE_INTEGER = 9007199254740991

# In-memory state:
# runId -> {
#   "input": canonical selection request,
#   "response": complete selection response
# }
runs = {}

TIME_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T"
    r"\d{2}:\d{2}:\d{2}"
    r"(?:\.\d{1,3})?"
    r"(?:Z|[+-]\d{2}:\d{2})$"
)

HEX64_RE = re.compile(r"^[0-9a-f]{64}$")


def utf8_key(s):
    return s.encode("utf-8")


def safe_int(x):
    return (
        isinstance(x, int)
        and not isinstance(x, bool)
        and 0 <= x <= MAX_SAFE_INTEGER
    )


def finite_number(x):
    return (
        isinstance(x, (int, float))
        and not isinstance(x, bool)
        and math.isfinite(float(x))
    )


def parse_time(value):
    if not isinstance(value, str):
        return None

    if not TIME_RE.fullmatch(value):
        return None

    # Validate the numeric offset explicitly.
    if not value.endswith("Z"):
        offset = value[-6:]

        try:
            oh = int(offset[1:3])
            om = int(offset[4:6])
        except ValueError:
            return None

        if oh > 14 or om > 59:
            return None

        if oh == 14 and om != 0:
            return None

    s = value[:-1] + "+00:00" if value.endswith("Z") else value

    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None

    if dt.tzinfo is None:
        return None

    try:
        offset = dt.utcoffset()
    except Exception:
        return None

    if offset is None:
        return None

    if abs(offset.total_seconds()) > 14 * 3600:
        return None

    return dt.astimezone(timezone.utc)


def utc_time_string(value):
    dt = parse_time(value)

    if dt is None:
        return None

    return (
        dt.strftime("%Y-%m-%dT%H:%M:%S.")
        + f"{dt.microsecond // 1000:03d}Z"
    )


def compact_json(obj):
    return json.dumps(
        obj,
        ensure_ascii=False,
        separators=(",", ":")
    )


def selection_fingerprint(data):
    # JSON object key ordering should not affect replay identity.
    return json.dumps(
        data,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True
    )


def dataset_digest(train_ids, eval_ids, feature_names):
    payload = {
        "trainRowIds": train_ids,
        "evalRowIds": eval_ids,
        "featureNames": feature_names
    }

    return hashlib.sha256(
        compact_json(payload).encode("utf-8")
    ).hexdigest()


def sort_codes(codes):
    return sorted(set(codes), key=utf8_key)


def invalid_http():
    return JSONResponse(
        status_code=400,
        content={"error": "INVALID_INPUT"}
    )


def invalid_select_response(run_id):
    return {
        "runId": run_id,
        "selectedTrialId": None,
        "trainRowIds": [],
        "evalRowIds": [],
        "featureNames": [],
        "datasetDigest": None,
        "reasonCodes": ["INVALID_INPUT"]
    }


# ==========================================================
# SELECT
# ==========================================================

def do_select(data):
    run_id = data.get("runId")
    forbidden = data.get("forbiddenFeatures")
    limit = data.get("numTrialsLimit")
    rows = data.get("rows")
    trials = data.get("trials")

    # --------------------------------------------------
    # Top-level validation
    # --------------------------------------------------

    if (
        not isinstance(run_id, str)
        or len(run_id) == 0
        or len(run_id) > 128
    ):
        return invalid_select_response(run_id)

    if (
        not isinstance(forbidden, list)
        or any(not isinstance(x, str) for x in forbidden)
    ):
        return invalid_select_response(run_id)

    # Positive integer -- NOT specified as safe integer.
    if (
        not isinstance(limit, int)
        or isinstance(limit, bool)
        or limit <= 0
    ):
        return invalid_select_response(run_id)

    if not isinstance(rows, list) or len(rows) == 0:
        return invalid_select_response(run_id)

    if not isinstance(trials, list):
        return invalid_select_response(run_id)

    # --------------------------------------------------
    # Validate selection rows
    # --------------------------------------------------

    expected_row_keys = {
        "id",
        "entity",
        "eventTime",
        "predictionTime",
        "version",
        "split",
        "features",
    }

    expected_feature_keys = {
        "value",
        "availableAt",
    }

    seen_ids = set()
    validated_rows = []

    for row in rows:
        if not isinstance(row, dict):
            return invalid_select_response(run_id)

        # Exact row shape
        if set(row.keys()) != expected_row_keys:
            return invalid_select_response(run_id)

        rid = row["id"]

        if not isinstance(rid, str):
            return invalid_select_response(run_id)

        if rid in seen_ids:
            return invalid_select_response(run_id)

        seen_ids.add(rid)

        if not isinstance(row["entity"], str):
            return invalid_select_response(run_id)

        event_dt = parse_time(row["eventTime"])
        prediction_dt = parse_time(row["predictionTime"])

        if event_dt is None or prediction_dt is None:
            return invalid_select_response(run_id)

        if not safe_int(row["version"]):
            return invalid_select_response(run_id)

        if row["split"] not in ("TRAIN", "EVAL"):
            return invalid_select_response(run_id)

        features = row["features"]

        if not isinstance(features, dict):
            return invalid_select_response(run_id)

        normalized_features = {}

        for feature_name, feature in features.items():
            if not isinstance(feature_name, str):
                return invalid_select_response(run_id)

            if not isinstance(feature, dict):
                return invalid_select_response(run_id)

            # Exact feature shape
            if set(feature.keys()) != expected_feature_keys:
                return invalid_select_response(run_id)

            available_dt = parse_time(feature["availableAt"])

            if available_dt is None:
                return invalid_select_response(run_id)

            # value is deliberately opaque data
            normalized_features[feature_name] = {
                "value": feature["value"],
                "availableAt": available_dt,
            }

        validated_rows.append({
            "id": rid,
            "entity": row["entity"],
            "eventTime": event_dt,
            "predictionTime": prediction_dt,
            "version": row["version"],
            "split": row["split"],
            "features": normalized_features,
        })

    # --------------------------------------------------
    # Deduplicate [entity, UTC(eventTime)]
    # --------------------------------------------------

    groups = {}

    for row in validated_rows:
        # datetime objects compare/hash by instant when timezone-aware,
        # but normalize explicitly to UTC for clarity.
        event_utc = row["eventTime"].astimezone(timezone.utc)

        key = (
            row["entity"],
            event_utc,
        )

        groups.setdefault(key, []).append(row)

    retained = []

    for candidates in groups.values():
        # highest version first,
        # then UTF-8-byte-smallest ID
        candidates.sort(
            key=lambda r: (
                -r["version"],
                utf8_key(r["id"]),
            )
        )

        retained.append(candidates[0])

    # --------------------------------------------------
    # Leakage-safe shared features
    # --------------------------------------------------

    common_features = set(retained[0]["features"].keys())

    for row in retained[1:]:
        common_features.intersection_update(
            row["features"].keys()
        )

    forbidden_set = set(forbidden)

    eligible_features = []

    for name in common_features:
        if name in forbidden_set:
            continue

        eligible = True

        for row in retained:
            available_at = row["features"][name]["availableAt"]
            prediction_time = row["predictionTime"]

            # Point-in-time condition is inclusive.
            if available_at > prediction_time:
                eligible = False
                break

        if eligible:
            eligible_features.append(name)

    feature_names = sorted(
        eligible_features,
        key=utf8_key,
    )

    # --------------------------------------------------
    # TRAIN / EVAL IDs after deduplication
    # --------------------------------------------------

    train_ids = sorted(
        [
            row["id"]
            for row in retained
            if row["split"] == "TRAIN"
        ],
        key=utf8_key,
    )

    eval_ids = sorted(
        [
            row["id"]
            for row in retained
            if row["split"] == "EVAL"
        ],
        key=utf8_key,
    )

    digest = dataset_digest(
        train_ids,
        eval_ids,
        feature_names,
    )

    # --------------------------------------------------
    # Trials
    # --------------------------------------------------

    expected_trial_keys = {
        "trialId",
        "status",
        "evalMetric",
    }

    seen_trial_ids = set()

    for trial in trials:
        if not isinstance(trial, dict):
            return invalid_select_response(run_id)

        if set(trial.keys()) != expected_trial_keys:
            return invalid_select_response(run_id)

        trial_id = trial["trialId"]

        if not safe_int(trial_id):
            return invalid_select_response(run_id)

        if trial_id in seen_trial_ids:
            return invalid_select_response(run_id)

        seen_trial_ids.add(trial_id)

        if trial["status"] not in ("SUCCEEDED", "FAILED"):
            return invalid_select_response(run_id)

    reason_codes = []

    if len(trials) > limit:
        reason_codes.append("TRIAL_LIMIT_EXCEEDED")

    successful = [
        trial
        for trial in trials
        if (
            trial["status"] == "SUCCEEDED"
            and finite_number(trial["evalMetric"])
        )
    ]

    if not successful:
        reason_codes.append("NO_SUCCESSFUL_TRIAL")

    selected_trial_id = None

    if not reason_codes:
        selected = min(
            successful,
            key=lambda trial: (
                -float(trial["evalMetric"]),
                trial["trialId"],
            ),
        )

        selected_trial_id = selected["trialId"]

    return {
        "runId": run_id,
        "selectedTrialId": selected_trial_id,
        "trainRowIds": train_ids,
        "evalRowIds": eval_ids,
        "featureNames": feature_names,
        "datasetDigest": digest,
        "reasonCodes": sort_codes(reason_codes),
        "datasetDigest": digest,
        "reasonCodes": sort_codes(reason_codes)
    }


# ==========================================================
# EVALUATE
# ==========================================================

def do_evaluate(data):

    run_id = data.get("runId")
    selected_trial = data.get("selectedTrialId")
    digest = data.get("datasetDigest")
    metric_floor = data.get("metricFloor")
    required_slices = data.get("requiredSlices")
    rows = data.get("rows")
    bytes_processed = data.get("bytesProcessed")
    max_bytes = data.get("maxBytes")

    codes = []

    # ---------- general input ----------

    input_valid = True

    if (
        not isinstance(run_id, str)
        or len(run_id) == 0
        or len(run_id) > 128
    ):
        input_valid = False

    if not safe_int(selected_trial):
        input_valid = False

    if (
        not isinstance(digest, str)
        or not HEX64_RE.fullmatch(digest)
    ):
        input_valid = False

    if (
        not finite_number(metric_floor)
        or not 0 <= float(metric_floor) <= 1
    ):
        input_valid = False

    if not isinstance(required_slices, dict):
        input_valid = False

    if not isinstance(rows, list):
        input_valid = False

    if not safe_int(bytes_processed):
        input_valid = False

    if not safe_int(max_bytes):
        input_valid = False

    if isinstance(required_slices, dict):
        for name, floor in required_slices.items():
            if (
                not isinstance(name, str)
                or len(name) == 0
                or not finite_number(floor)
                or not 0 <= float(floor) <= 1
            ):
                input_valid = False

    if not input_valid:
        codes.append("INVALID_INPUT")

    # ---------- lineage ----------

    lineage_valid = False

    stored = runs.get(run_id)

    if stored is not None:

        selection = stored["response"]

        if (
            selection["reasonCodes"] == []
            and selection["selectedTrialId"] is not None
            and selected_trial == selection["selectedTrialId"]
            and digest == selection["datasetDigest"]
        ):
            lineage_valid = True

    if not lineage_valid:
        codes.append("INVALID_LINEAGE")

    # ---------- bytes ----------
    #
    # This check still runs even when rows are invalid.

    if (
        safe_int(bytes_processed)
        and safe_int(max_bytes)
        and bytes_processed > max_bytes
    ):
        codes.append("BYTE_LIMIT")

    # ---------- test rows ----------

    test_rows_valid = (
        isinstance(rows, list)
        and len(rows) > 0
    )

    if isinstance(rows, list):

        for row in rows:

            if not isinstance(row, dict):
                test_rows_valid = False
                break

            if not {
                "label",
                "prediction",
                "slice"
            }.issubset(row.keys()):
                test_rows_valid = False
                break

            label = row["label"]
            prediction = row["prediction"]
            slice_name = row["slice"]

            if (
                not isinstance(label, int)
                or isinstance(label, bool)
                or label not in (0, 1)
            ):
                test_rows_valid = False
                break

            if (
                not isinstance(prediction, int)
                or isinstance(prediction, bool)
                or prediction not in (0, 1)
            ):
                test_rows_valid = False
                break

            if (
                not isinstance(slice_name, str)
                or len(slice_name) == 0
            ):
                test_rows_valid = False
                break

    if not test_rows_valid:
        codes.append("INVALID_TEST_ROW")

    test_metric = None

    # false for invalid input / lineage / test row
    critical_slice_pass = (
        input_valid
        and lineage_valid
        and test_rows_valid
    )

    # Only perform aggregate/slice checks when every test row is valid.
    if test_rows_valid and isinstance(required_slices, dict):

        total_correct = sum(
            1
            for row in rows
            if row["label"] == row["prediction"]
        )

        test_metric = round(
            total_correct / len(rows),
            12
        )

        if (
            finite_number(metric_floor)
            and test_metric < float(metric_floor)
        ):
            codes.append("AGGREGATE_FLOOR")

        for slice_name, floor in required_slices.items():

            # Invalid required-slice definitions are handled by
            # INVALID_INPUT; don't generate fake slice codes.
            if (
                not isinstance(slice_name, str)
                or len(slice_name) == 0
                or not finite_number(floor)
                or not 0 <= float(floor) <= 1
            ):
                critical_slice_pass = False
                continue

            subset = [
                r for r in rows
                if r["slice"] == slice_name
            ]

            if not subset:
                codes.append(
                    f"MISSING_SLICE:{slice_name}"
                )
                critical_slice_pass = False
                continue

            correct = sum(
                1
                for r in subset
                if r["label"] == r["prediction"]
            )

            accuracy = round(
                correct / len(subset),
                12
            )

            if accuracy < float(floor):
                codes.append(
                    f"SLICE_FLOOR:{slice_name}"
                )
                critical_slice_pass = False

    codes = sort_codes(codes)

    decision = "admit" if not codes else "reject"

    return {
        "runId": run_id,
        "selectedTrialId": (
            selected_trial
            if safe_int(selected_trial)
            else None
        ),
        "datasetDigest": digest,
        "testMetric": test_metric,
        "criticalSlicePass": critical_slice_pass,
        "decision": decision,
        "bytesProcessed": bytes_processed,
        "reasonCodes": codes
    }


# ==========================================================
# ENDPOINT
# ==========================================================

@app.post("/bqml")
async def bqml(request: Request):

    # Avoid FastAPI's automatic 422 response.
    # The assignment wants INVALID_INPUT behavior instead.
    try:
        data = await request.json()
    except Exception:
        return invalid_http()

    if not isinstance(data, dict):
        return invalid_http()

    phase = data.get("phase")

    if phase not in ("select", "evaluate"):
        return invalid_http()

    if phase == "evaluate":
        return do_evaluate(data)

    # ======================================================
    # Stateful SELECT / replay
    # ======================================================

    run_id = data.get("runId")

    # Invalid runId can't safely participate in persistence.
    if (
        not isinstance(run_id, str)
        or len(run_id) == 0
        or len(run_id) > 128
    ):
        return do_select(data)

    fingerprint = selection_fingerprint(data)

    if run_id in runs:

        if runs[run_id]["input"] == fingerprint:
            return copy.deepcopy(runs[run_id]["response"])

        return JSONResponse(
            status_code=409,
            content={"error": "RUN_ID_CONFLICT"}
        )

    response = do_select(data)

    # Persist complete selection response under runId,
    # including failed/malformed selections with a valid runId.
    runs[run_id] = {
        "input": fingerprint,
        "response": copy.deepcopy(response)
    }

    return response