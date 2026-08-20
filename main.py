import hashlib
import json
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, Response, status
from pydantic import BaseModel, Field

app = FastAPI(title="BQML Experiment Gate")

# Stateful storage mock (in production, use Redis or a database)
STORAGE: Dict[str, Dict[str, Any]] = {}


class BQMLRequest(BaseModel):
    phase: str
    runId: str = Field(..., max_length=128)
    forbiddenFeatures: Optional[List[str]] = []
    numTrialsLimit: Optional[int] = 10
    rows: Optional[List[Dict[str, Any]]] = []
    trials: Optional[List[Dict[str, Any]]] = []
    selectedTrialId: Optional[int] = None
    datasetDigest: Optional[str] = None
    metricFloor: Optional[float] = 0.8
    requiredSlices: Optional[Dict[str, float]] = {}
    bytesProcessed: Optional[int] = 0
    maxBytes: Optional[int] = 2000


@app.post("/bqml")
async def handle_bqml(payload: BQMLRequest, response: Response):
    phase = payload.phase
    run_id = payload.runId

    if phase not in ("select", "evaluate"):
        return Response(
            content=json.dumps({"error": "INVALID_INPUT"}),
            status_code=400,
            media_type="application/json",
        )

    # ==========================================
    # PHASE 1: MODEL SELECTION
    # ==========================================
    if phase == "select":
        reason_codes = []

        # Check trial limits
        if payload.trials and len(payload.trials) > (payload.numTrialsLimit or 10):
            reason_codes.append("TRIAL_LIMIT_EXCEEDED")

        # Deduplicate rows by [entity, UTC(eventTime)]
        # Keep highest integer version, then UTF-8-byte-smallest ID
        dedup_map = {}
        for row in payload.rows or []:
            key = (row.get("entity"), row.get("eventTime"))
            ver = row.get("version", 1)
            rid = row.get("id", "")
            if key not in dedup_map:
                dedup_map[key] = row
            else:
                curr = dedup_map[key]
                curr_ver = curr.get("version", 1)
                curr_id = curr.get("id", "")
                if ver > curr_ver or (ver == curr_ver and rid < curr_id):
                    dedup_map[key] = row

        retained_rows = list(dedup_map.values())
        train_row_ids = sorted(
            [r["id"] for r in retained_rows if r.get("split") == "TRAIN"],
            key=lambda x: x.encode("utf-8"),
        )
        eval_row_ids = sorted(
            [r["id"] for r in retained_rows if r.get("split") == "EVAL"],
            key=lambda x: x.encode("utf-8"),
        )

        # Feature eligibility check
        all_feature_names = set()
        for r in retained_rows:
            for fname, fval in r.get("features", {}).items():
                all_feature_names.add(fname)

        eligible_features = []
        for fname in sorted(all_feature_names, key=lambda x: x.encode("utf-8")):
            if fname in (payload.forbiddenFeatures or []):
                continue
            is_eligible = True
            for r in retained_rows:
                feats = r.get("features", {})
                if fname not in feats:
                    is_eligible = False
                    break
                # Check availableAt <= predictionTime
                if feats[fname].get("availableAt", "") > r.get("predictionTime", ""):
                    is_eligible = False
                    break
            if is_eligible:
                eligible_features.append(fname)

        # Trial filtering & optimization
        successful_trials = [
            t
            for t in (payload.trials or [])
            if t.get("status") == "SUCCEEDED"
            and isinstance(t.get("evalMetric"), (int, float))
        ]

        selected_trial_id = None
        if not successful_trials or "TRIAL_LIMIT_EXCEEDED" in reason_codes:
            reason_codes.append("NO_SUCCESSFUL_TRIAL")
            selected_trial_id = None
        else:
            # Maximize evalMetric, break ties with smallest integer trialId
            best_trial = max(
                successful_trials,
                key=lambda t: (t["evalMetric"], -t["trialId"]),
            )
            selected_trial_id = best_trial["trialId"]

        # Compute datasetDigest via compact JSON SHA-256
        digest_dict = {
            "trainRowIds": train_row_ids,
            "evalRowIds": eval_row_ids,
            "featureNames": eligible_features,
        }
        compact_json = json.dumps(
            digest_dict, separators=(",", ":"), sort_keys=False
        )
        dataset_digest = hashlib.sha256(compact_json.encode("utf-8")).hexdigest()

        if reason_codes:
            selected_trial_id = None

        result = {
            "runId": run_id,
            "selectedTrialId": selected_trial_id,
            "trainRowIds": train_row_ids,
            "evalRowIds": eval_row_ids,
            "featureNames": eligible_features,
            "datasetDigest": dataset_digest,
            "reasonCodes": sorted(list(set(reason_codes)), key=lambda x: x.encode("utf-8")),
        }

        # Idempotency checks
        if run_id in STORAGE:
            stored_data = STORAGE[run_id]
            if stored_data != result:
                return Response(
                    content=json.dumps({"error": "RUN_ID_CONFLICT"}),
                    status_code=409,
                    media_type="application/json",
                )
        else:
            STORAGE[run_id] = result

        return result

    # ==========================================
    # PHASE 2: FROZEN TRIAL EVALUATION
    # ==========================================
    elif phase == "evaluate":
        eval_reasons = []
        stored = STORAGE.get(run_id)

        # Lineage verification
        if not stored or stored.get("selectedTrialId") != payload.selectedTrialId or stored.get("datasetDigest") != payload.datasetDigest:
            eval_reasons.append("INVALID_LINEAGE")

        # Byte limits check
        if (payload.bytesProcessed or 0) > (payload.maxBytes or 0):
            eval_reasons.append("BYTE_LIMIT")

        rows = payload.rows or []
        invalid_row_found = False
        slice_correct_counts: Dict[str, int] = {}
        slice_total_counts: Dict[str, int] = {}
        total_correct = 0

        if not rows:
            invalid_row_found = True
            eval_reasons.append("INVALID_TEST_ROW")
        else:
            for r in rows:
                label = r.get("label")
                pred = r.get("prediction")
                s = r.get("slice")
                if label not in (0, 1) or pred not in (0, 1) or not s:
                    invalid_row_found = True
                    eval_reasons.append("INVALID_TEST_ROW")
                    continue
                
                if s not in slice_total_counts:
                    slice_total_counts[s] = 0
                    slice_correct_counts[s] = 0
                
                slice_total_counts[s] += 1
                if label == pred:
                    slice_correct_counts[s] += 1
                    total_correct += 1

        test_metric = None
        if rows and not invalid_row_found:
            test_metric = round(total_correct / len(rows), 12)

        # Check required slices
        required_slices = payload.requiredSlices or {}
        missing_slices = [sname for sname in required_slices if sname not in slice_total_counts]
        for ms in missing_slices:
            eval_reasons.append(f"MISSING_SLICE:{ms}")

        slice_floors_failed = []
        for sname, floor in required_slices.items():
            if sname in slice_total_counts and slice_total_counts[sname] > 0:
                s_acc = slice_correct_counts[sname] / slice_total_counts[sname]
                if s_acc < floor:
                    slice_floors_failed.append(sname)
                    eval_reasons.append(f"SLICE_FLOOR:{sname}")

        # Aggregate floor check
        if test_metric is not None and test_metric < (payload.metricFloor or 0.8):
            eval_reasons.append("AGGREGATE_FLOOR")

        # Determine criticalSlicePass
        critical_pass = True
        if eval_reasons or invalid_row_found or missing_slices or slice_floors_failed:
            critical_pass = False

        decision = "admit" if not eval_reasons and critical_pass else "reject"

        # Unique, sorted reason codes by UTF-8 bytes
        sorted_reasons = sorted(list(set(eval_reasons)), key=lambda x: x.encode("utf-8"))

        return {
            "runId": run_id,
            "selectedTrialId": payload.selectedTrialId,
            "datasetDigest": payload.datasetDigest,
            "testMetric": test_metric,
            "criticalSlicePass": critical_pass,
            "decision": decision,
            "bytesProcessed": payload.bytesProcessed,
            "reasonCodes": sorted_reasons,
        }