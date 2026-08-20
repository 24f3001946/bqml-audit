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

    if seconds > 14 * 3600:
        return None

    # +14:00/-14:00 allowed, but not 14:xx
    if seconds == 14 * 3600:
        raw_offset = value[-6:] if not value.endswith("Z") else "+00:00"
        if raw_offset[-2:] != "00":
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

    # ---------- top-level selection validation ----------

    malformed = False

    if (
        not isinstance(run_id, str)
        or len(run_id) == 0
        or len(run_id) > 128
    ):
        malformed = True

    if (
        not isinstance(forbidden, list)
        or any(not isinstance(x, str) for x in forbidden)
    ):
        malformed = True

    if not safe_int(limit) or limit == 0:
        malformed = True

    if not isinstance(rows, list) or len(rows) == 0:
        malformed = True

    if not isinstance(trials, list):
        malformed = True

    if malformed:
        return invalid_select_response(run_id)

    # ---------- validate all original rows ----------

    seen_row_ids = set()
    valid_rows = []

    for row in rows:

        if not isinstance(row, dict):
            return invalid_select_response(run_id)

        required = {
            "id",
            "entity",
            "eventTime",
            "predictionTime",
            "version",
            "split",
            "features"
        }

        # Required fields must exist.
        if not required.issubset(row.keys()):
            return invalid_select_response(run_id)

        rid = row["id"]

        if not isinstance(rid, str):
            return invalid_select_response(run_id)

        if rid in seen_row_ids:
            return invalid_select_response(run_id)

        seen_row_ids.add(rid)

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

            if "value" not in feature or "availableAt" not in feature:
                return invalid_select_response(run_id)

            available_dt = parse_time(feature["availableAt"])

            if available_dt is None:
                return invalid_select_response(run_id)

            # Feature value is deliberately treated only as data.
            normalized_features[feature_name] = {
                "value": feature["value"],
                "availableAt": available_dt
            }

        valid_rows.append({
            "id": rid,
            "entity": row["entity"],
            "event_dt": event_dt,
            "prediction_dt": prediction_dt,
            "version": row["version"],
            "split": row["split"],
            "features": normalized_features
        })

    # ---------- deduplicate ----------
    #
    # Tuple is [entity, UTC(eventTime)].
    # Keep highest version, then UTF-8-smallest ID.

    groups = {}

    for row in valid_rows:
        event_utc = (
            row["event_dt"]
            .astimezone(timezone.utc)
            .isoformat()
        )

        key = (row["entity"], event_utc)

        groups.setdefault(key, []).append(row)

    retained = []

    for group in groups.values():

        group.sort(
            key=lambda r: (
                -r["version"],
                utf8_key(r["id"])
            )
        )

        retained.append(group[0])

    # ---------- shared leakage-safe features ----------

    if not retained:
        return invalid_select_response(run_id)

    common_names = set(retained[0]["features"].keys())

    for row in retained[1:]:
        common_names &= set(row["features"].keys())

    forbidden_set = set(forbidden)

    feature_names = []

    for name in common_names:

        if name in forbidden_set:
            continue

        point_in_time_safe = True

        for row in retained:
            available_dt = row["features"][name]["availableAt"]
            prediction_dt = row["prediction_dt"]

            if available_dt > prediction_dt:
                point_in_time_safe = False
                break

        if point_in_time_safe:
            feature_names.append(name)

    feature_names.sort(key=utf8_key)

    # ---------- split IDs ----------

    train_ids = sorted(
        [r["id"] for r in retained if r["split"] == "TRAIN"],
        key=utf8_key
    )

    eval_ids = sorted(
        [r["id"] for r in retained if r["split"] == "EVAL"],
        key=utf8_key
    )

    digest = dataset_digest(
        train_ids,
        eval_ids,
        feature_names
    )

    # ---------- validate trials ----------

    seen_trial_ids = set()

    for trial in trials:

        if not isinstance(trial, dict):
            return invalid_select_response(run_id)

        if not {
            "trialId",
            "status",
            "evalMetric"
        }.issubset(trial.keys()):
            return invalid_select_response(run_id)

        tid = trial["trialId"]

        if not safe_int(tid):
            return invalid_select_response(run_id)

        if tid in seen_trial_ids:
            return invalid_select_response(run_id)

        seen_trial_ids.add(tid)

        if trial["status"] not in ("SUCCEEDED", "FAILED"):
            return invalid_select_response(run_id)

    reason_codes = []

    if len(trials) > limit:
        reason_codes.append("TRIAL_LIMIT_EXCEEDED")

    eligible_trials = [
        t for t in trials
        if (
            t["status"] == "SUCCEEDED"
            and finite_number(t["evalMetric"])
        )
    ]

    if not eligible_trials:
        reason_codes.append("NO_SUCCESSFUL_TRIAL")

    selected_id = None

    if not reason_codes:

        # Highest metric.
        # Exact tie -> smallest integer trialId.
        selected = min(
            eligible_trials,
            key=lambda t: (
                -float(t["evalMetric"]),
                t["trialId"]
            )
        )

        selected_id = selected["trialId"]

    return {
        "runId": run_id,
        "selectedTrialId": selected_id,
        "trainRowIds": train_ids,
        "evalRowIds": eval_ids,
        "featureNames": feature_names,

        # IMPORTANT:
        # Dataset is valid even when trial limit/no-successful-trial
        # occurs. Only malformed selections get null digest.
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