"""Automated tests, consistency checks, and latency/VRAM benchmarks for DermaScan Face AI.

Suite conforms to task_w5_tests_and_benchmarks.md:
1. test_health_endpoint: Verifies /api/health returns 200 and expected status/calibration payload.
2. test_static_frontend_serving: Verifies static files and index.html serving.
3. test_valid_selfie_and_contract_keys: Verifies full-face selfie and 21 frozen contract keys.
4. test_valid_closeup_and_triage_mode: Verifies closeup image and triage mode prior shift.
5. test_all_invalid_inputs_and_error_states: Verifies 6 input error states and friendly codes.
6. test_consistency_against_test_predictions_csv: Verifies 16 test images against CSV probabilities.
7. test_eight_demo_cases_and_benchmark_summary: Verifies 8 demo cases and prints latency/VRAM summary.
"""
from __future__ import annotations

import base64
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import pandas as pd
import pytest
import torch
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.calibration import predict_calibrated_ensemble
from src.dataset import CLASS_NAMES
from webapp.app import app
from webapp.service import (
    ABSTENTION_THRESHOLD,
    CALIBRATION_T_EFFNET,
    CALIBRATION_T_RESNET,
    CALIBRATION_W_EFFNET,
    CALIBRATION_W_RESNET,
    MAX_UPLOAD_BYTES,
)

FROZEN_CONTRACT_KEYS = {
    "prediction",
    "prediction_label",
    "confidence",
    "abstained",
    "abstention_threshold",
    "abstention_message",
    "caution_note",
    "triage_mode",
    "probabilities",
    "top2",
    "input_mode",
    "face_detected",
    "num_patches",
    "dominant_region",
    "region_probabilities",
    "model_breakdown",
    "original_image",
    "gradcam_image",
    "latency_ms",
    "info",
    "disclaimer",
}


@pytest.fixture(scope="module")
def client():
    """Module-scoped TestClient to avoid reloading model weights for each test."""
    with TestClient(app) as c:
        yield c


def _load_merged_test_manifest() -> pd.DataFrame:
    """Load test predictions merged with manifest metadata for split == 'test'."""
    pred_path = PROJECT_ROOT / "runs" / "ensemble_seed42" / "test_predictions.csv"
    man_path = PROJECT_ROOT / "data" / "manifest" / "manifest.csv"

    df_pred = pd.read_csv(pred_path)
    df_man = pd.read_csv(man_path)

    merged = pd.merge(
        df_pred,
        df_man[["image_id", "path", "face_detected", "split"]],
        on="image_id",
        how="inner",
    )
    assert len(merged) == 303, f"Expected 303 test rows, found {len(merged)}"
    return merged


def _compute_calibrated_probs(df: pd.DataFrame) -> np.ndarray:
    """Compute calibrated ensemble probabilities for all rows in test dataframe."""
    res_cols = [f"resnet_prob_{c}" for c in CLASS_NAMES]
    eff_cols = [f"effnet_prob_{c}" for c in CLASS_NAMES]
    res_probs = df[res_cols].values
    eff_probs = df[eff_cols].values

    return predict_calibrated_ensemble(
        res_probs,
        eff_probs,
        w_resnet=CALIBRATION_W_RESNET,
        t_resnet=CALIBRATION_T_RESNET,
        t_effnet=CALIBRATION_T_EFFNET,
    )


# ==============================================================================
# 1. Health Endpoint Test
# ==============================================================================

def test_health_endpoint(client: TestClient):
    """Verify GET /api/health returns HTTP 200 with model status and calibration constants."""
    resp = client.get("/api/health")
    assert resp.status_code == 200, f"Health endpoint failed: {resp.status_code}"

    data = resp.json()
    assert data["status"] == "ok"
    assert data["models_loaded"] is True
    assert data["knowledge_base_loaded"] is True
    assert data["abstention_threshold"] == ABSTENTION_THRESHOLD
    assert data["calibration"] == {
        "t_resnet": CALIBRATION_T_RESNET,
        "t_effnet": CALIBRATION_T_EFFNET,
        "w_resnet": CALIBRATION_W_RESNET,
        "w_effnet": CALIBRATION_W_EFFNET,
    }


# ==============================================================================
# 2. Static Frontend Serving Test
# ==============================================================================

def test_static_frontend_serving(client: TestClient):
    """Verify static frontend assets and root single-page application load successfully."""
    static_routes = [
        "/",
        "/style.css",
        "/app.js",
        "/logo.svg",
        "/static/mock_result.json",
    ]
    for route in static_routes:
        resp = client.get(route)
        assert resp.status_code == 200, f"Route {route} failed with status {resp.status_code}"
        assert len(resp.content) > 0, f"Route {route} returned empty body"

    resp_root = client.get("/")
    html_text = resp_root.text
    assert '<meta charset="utf-8">' in html_text.lower()
    assert "DermaScan Face AI" in html_text


# ==============================================================================
# 3. Valid Selfie & Frozen Contract Keys Test
# ==============================================================================

def test_valid_selfie_and_contract_keys(client: TestClient):
    """Verify full-face selfie analysis conforms strictly to the 21 frozen contract keys."""
    merged = _load_merged_test_manifest()
    # Choose a full-face selfie from test split
    face_rows = merged[(merged["split"] == "test") & (merged["face_detected"] == True)]
    assert len(face_rows) > 0, "No face_detected images found in test split"

    sample = face_rows.iloc[0]
    img_path = PROJECT_ROOT / sample["path"]
    assert img_path.exists(), f"Image not found at {img_path}"

    with open(img_path, "rb") as f:
        file_bytes = f.read()

    resp = client.post(
        "/api/analyze",
        files={"file": (img_path.name, file_bytes, "image/jpeg")},
    )
    assert resp.status_code == 200, f"POST /api/analyze failed: {resp.status_code} {resp.text}"

    data = resp.json()

    # 1. Verify all 21 frozen contract keys exist
    missing_keys = FROZEN_CONTRACT_KEYS - set(data.keys())
    assert not missing_keys, f"Missing frozen contract keys: {missing_keys}"

    # 2. Verify face mode properties
    assert data["input_mode"] == "face"
    assert data["face_detected"] is True
    assert data["num_patches"] == 5
    expected_regions = {"forehead", "nose", "left_cheek", "right_cheek", "chin"}
    assert set(data["region_probabilities"].keys()) == expected_regions

    # 3. Verify probabilities sum to 1.0 within numerical tolerance
    probs = data["probabilities"]
    assert abs(sum(probs.values()) - 1.0) < 1e-3, f"Probabilities do not sum to 1.0: {probs}"

    # 4. Verify base64 images
    for img_key in ("original_image", "gradcam_image"):
        val = data[img_key]
        assert val.startswith("data:image/png;base64,"), f"{img_key} does not have PNG data URI header"
        raw_b64 = val.split(",", 1)[1]
        decoded = base64.b64decode(raw_b64)
        assert len(decoded) > 0, f"{img_key} decoded to empty bytes"

    # 5. Verify knowledge base info contains en, th, sources and Thai characters
    info = data["info"]
    assert "en" in info and "th" in info and "sources" in info
    th_name = info["th"].get("name", "")
    assert bool(th_name), "Thai name is missing or empty"
    has_thai_chars = any("\u0e00" <= ch <= "\u0e7f" for ch in th_name)
    assert has_thai_chars, f"Expected Thai characters in th['name'], got: {th_name}"


# ==============================================================================
# 4. Valid Close-up & Triage Mode Test
# ==============================================================================

def test_valid_closeup_and_triage_mode(client: TestClient):
    """Verify close-up skin photo (no face) with triage mode re-weights symptomatic priors."""
    merged = _load_merged_test_manifest()
    # Find a test close-up with skin detected (e.g. roboflow_rosacea_70352d186839)
    closeup_sample = merged[merged["image_id"] == "roboflow_rosacea_70352d186839"].iloc[0]
    img_path = PROJECT_ROOT / closeup_sample["path"]
    assert img_path.exists(), f"Image not found at {img_path}"

    with open(img_path, "rb") as f:
        file_bytes = f.read()

    resp = client.post(
        "/api/analyze?triage_mode=true",
        files={"file": (img_path.name, file_bytes, "image/jpeg")},
    )
    assert resp.status_code == 200, f"POST /api/analyze failed: {resp.status_code} {resp.text}"

    data = resp.json()

    # Verify closeup mode attributes
    assert data["input_mode"] == "closeup"
    assert data["face_detected"] is False
    assert 1 <= data["num_patches"] <= 9
    assert data["region_probabilities"] is None

    # Verify triage mode attributes: normal must be exactly 0.0
    assert data["triage_mode"] is True
    assert data["probabilities"]["normal"] == 0.0
    assert abs(sum(data["probabilities"].values()) - 1.0) < 1e-3


# ==============================================================================
# 5. Invalid Inputs & Error States Test
# ==============================================================================

def test_all_invalid_inputs_and_error_states(client: TestClient):
    """Verify 6 input validation error scenarios and friendly bilingual error responses."""
    # 1. No file provided
    r1 = client.post("/api/analyze")
    assert r1.status_code == 400
    assert r1.json()["error_code"] == "empty_file"

    # 2. Empty 0-byte file
    r2 = client.post(
        "/api/analyze",
        files={"file": ("empty.jpg", b"", "image/jpeg")},
    )
    assert r2.status_code == 400
    assert r2.json()["error_code"] == "empty_file"

    # 3. Unsupported file extension
    r3 = client.post(
        "/api/analyze",
        files={"file": ("document.pdf", b"%PDF-1.4 dummy", "application/pdf")},
    )
    assert r3.status_code == 415
    assert r3.json()["error_code"] == "invalid_file_type"

    # 4. Corrupted image bytes
    r4 = client.post(
        "/api/analyze",
        files={"file": ("corrupt.jpg", b"not-a-valid-jpeg", "image/jpeg")},
    )
    assert r4.status_code == 400
    assert r4.json()["error_code"] == "unreadable_image"

    # 5. Oversized file (> 10 MB)
    big_bytes = b"0" * (MAX_UPLOAD_BYTES + 100)
    r5 = client.post(
        "/api/analyze",
        files={"file": ("big.jpg", big_bytes, "image/jpeg")},
    )
    assert r5.status_code == 413
    assert r5.json()["error_code"] == "file_too_large"

    # 6. No face & no usable skin region (Case 8: solid blue non-skin image)
    blue_img = np.zeros((300, 300, 3), dtype=np.uint8)
    blue_img[:, :] = (255, 0, 0)  # BGR solid blue
    success, encoded_blue = cv2.imencode(".png", blue_img)
    assert success

    r6 = client.post(
        "/api/analyze",
        files={"file": ("noskin.png", encoded_blue.tobytes(), "image/png")},
    )
    assert r6.status_code == 422
    data6 = r6.json()
    assert data6["error_code"] == "no_skin_detected"
    assert bool(data6.get("detail")) and bool(data6.get("detail_th"))


# ==============================================================================
# 6. Consistency Against test_predictions.csv (16 Test Images)
# ==============================================================================

def test_consistency_against_test_predictions_csv(client: TestClient):
    """Verify live API predictions against calibrated test_predictions.csv for 16 images."""
    merged = _load_merged_test_manifest()
    cal_probs_all = _compute_calibrated_probs(merged)

    merged["cal_pred"] = [CLASS_NAMES[i] for i in np.argmax(cal_probs_all, axis=1)]
    merged["cal_conf"] = np.max(cal_probs_all, axis=1)

    # Select 16 representative images: 4 correctly classified high-confidence per class
    selected_list = []
    for c in CLASS_NAMES:
        subset = merged[(merged["true_label"] == c) & (merged["cal_pred"] == c)].sort_values(
            "cal_conf", ascending=False
        )
        selected_list.append(subset.head(4))

    selected_df = pd.concat(selected_list, ignore_index=True)
    assert len(selected_df) == 16, f"Expected 16 images, selected {len(selected_df)}"

    top1_matches = 0
    max_diffs: List[float] = []

    for idx, row in selected_df.iterrows():
        img_p = PROJECT_ROOT / row["path"]
        assert img_p.exists(), f"Image path does not exist: {img_p}"

        with open(img_p, "rb") as f:
            b = f.read()

        resp = client.post(
            "/api/analyze",
            files={"file": (img_p.name, b, "image/jpeg")},
        )
        assert resp.status_code == 200, f"Failed on {row['image_id']}: {resp.status_code}"
        res_json = resp.json()

        # Find original calibrated probabilities from CSV for this exact image_id
        orig_idx = merged[merged["image_id"] == row["image_id"]].index[0]
        expected_cal = cal_probs_all[orig_idx]

        api_probs = np.array([res_json["probabilities"][c.lower()] for c in CLASS_NAMES])
        diff = float(np.max(np.abs(api_probs - expected_cal)))
        max_diffs.append(diff)

        if res_json["prediction_label"] == row["cal_pred"]:
            top1_matches += 1

    agreement_fraction = top1_matches / 16.0
    mean_abs_diff = float(np.mean(max_diffs))

    print(f"\n[Consistency Check] Top-1 Agreement: {top1_matches}/16 ({agreement_fraction * 100:.2f}%)")
    print(f"[Consistency Check] Mean Max Absolute Diff: {mean_abs_diff:.4f} (Max: {max(max_diffs):.4f})")

    # Assertions per spec
    assert top1_matches >= 15, f"Expected >= 15/16 top-1 matches, got {top1_matches}/16"
    assert mean_abs_diff < 0.03, f"Expected mean max diff < 0.03, got {mean_abs_diff:.4f}"


# ==============================================================================
# 7. Eight Demo Cases & Benchmark Summary
# ==============================================================================

def test_eight_demo_cases_and_benchmark_summary(client: TestClient):
    """Verify all 8 demo cases and print a clean benchmark summary table with latency & VRAM."""
    merged = _load_merged_test_manifest()
    cal_probs_all = _compute_calibrated_probs(merged)
    merged["cal_pred"] = [CLASS_NAMES[i] for i in np.argmax(cal_probs_all, axis=1)]
    merged["cal_conf"] = np.max(cal_probs_all, axis=1)

    # 1. Identify Demo Cases 1 to 6
    # Case 1: Full-face selfie -> Acne (true & pred Acne, conf >= threshold)
    c1_row = merged[
        (merged["face_detected"] == True)
        & (merged["true_label"] == "Acne")
        & (merged["cal_pred"] == "Acne")
        & (merged["cal_conf"] >= ABSTENTION_THRESHOLD)
    ].sort_values("cal_conf", ascending=False).iloc[0]

    # Case 2: Full-face selfie -> Rosacea (true & pred Rosacea, conf >= threshold)
    c2_row = merged[
        (merged["face_detected"] == True)
        & (merged["true_label"] == "Rosacea")
        & (merged["cal_pred"] == "Rosacea")
        & (merged["cal_conf"] >= ABSTENTION_THRESHOLD)
    ].sort_values("cal_conf", ascending=False).iloc[0]

    # Case 3: Dermatitis case (true & pred Dermatitis, conf >= threshold)
    c3_row = merged[
        (merged["true_label"] == "Dermatitis")
        & (merged["cal_pred"] == "Dermatitis")
        & (merged["cal_conf"] >= ABSTENTION_THRESHOLD)
    ].sort_values("cal_conf", ascending=False).iloc[0]

    # Case 4: Normal face (true & pred Normal, face_detected True, conf >= threshold)
    c4_row = merged[
        (merged["face_detected"] == True)
        & (merged["true_label"] == "Normal")
        & (merged["cal_pred"] == "Normal")
        & (merged["cal_conf"] >= ABSTENTION_THRESHOLD)
    ].sort_values("cal_conf", ascending=False).iloc[0]

    # Case 5: Close-up (face_detected False, correctly classified)
    c5_row = merged[
        (merged["face_detected"] == False)
        & (merged["true_label"] == merged["cal_pred"])
        & (merged["cal_conf"] >= 0.85)
    ].sort_values("cal_conf", ascending=False).iloc[0]

    # Case 6: Low confidence image (cal_conf < ABSTENTION_THRESHOLD)
    c6_row = merged[merged["cal_conf"] < ABSTENTION_THRESHOLD].sort_values("cal_conf").iloc[0]

    demo_cases_meta = [
        ("Case 1 (Full-face selfie -> Acne)", c1_row),
        ("Case 2 (Full-face selfie -> Rosacea)", c2_row),
        ("Case 3 (Dermatitis case)", c3_row),
        ("Case 4 (Normal face)", c4_row),
        ("Case 5 (Close-up, no face detected)", c5_row),
        ("Case 6 (Low-confidence -> Abstention)", c6_row),
    ]

    demo_results = []
    latencies: List[float] = []

    # Execute Demo Cases 1 to 6
    for desc, row in demo_cases_meta:
        img_p = PROJECT_ROOT / row["path"]
        assert img_p.exists(), f"Image not found at {img_p}"

        with open(img_p, "rb") as f:
            b = f.read()

        resp = client.post(
            "/api/analyze",
            files={"file": (img_p.name, b, "image/jpeg")},
        )
        assert resp.status_code == 200, f"Demo case '{desc}' failed with HTTP {resp.status_code}"
        res = resp.json()
        latencies.append(res["latency_ms"])

        demo_results.append({
            "desc": desc,
            "image_id": row["image_id"],
            "path": row["path"],
            "input_mode": res["input_mode"],
            "num_patches": res["num_patches"],
            "prediction": res["prediction"],
            "confidence": res["confidence"],
            "abstained": res["abstained"],
            "top2": res["top2"],
            "latency_ms": res["latency_ms"],
            "caution_note": res["caution_note"],
            "abstention_message": res["abstention_message"],
        })

    # Verify specific assertions for each case
    # Case 1: Acne, not abstained
    assert demo_results[0]["prediction"] == "acne"
    assert demo_results[0]["abstained"] is False
    assert demo_results[0]["input_mode"] == "face"

    # Case 2: Rosacea, not abstained
    assert demo_results[1]["prediction"] == "rosacea"
    assert demo_results[1]["abstained"] is False
    assert demo_results[1]["input_mode"] == "face"

    # Case 3: Dermatitis, not abstained, caution note present
    assert demo_results[2]["prediction"] == "dermatitis"
    assert demo_results[2]["abstained"] is False
    assert demo_results[2]["caution_note"] is not None

    # Case 4: Normal, not abstained, caution note present
    assert demo_results[3]["prediction"] == "normal"
    assert demo_results[3]["abstained"] is False
    assert demo_results[3]["caution_note"] is not None

    # Case 5: Close-up mode, patches >= 1
    assert demo_results[4]["input_mode"] == "closeup"
    assert demo_results[4]["num_patches"] >= 1

    # Case 6: Abstained is True, non-empty message with top2
    assert demo_results[5]["abstained"] is True
    assert demo_results[5]["abstention_message"] is not None
    assert len(demo_results[5]["top2"]) == 2

    # Case 7: Invalid input (unsupported extension / corrupted / too large)
    r7_ext = client.post(
        "/api/analyze",
        files={"file": ("invalid.txt", b"hello", "text/plain")},
    )
    assert r7_ext.status_code == 415

    r7_corrupt = client.post(
        "/api/analyze",
        files={"file": ("corrupt.jpg", b"corrupted-data", "image/jpeg")},
    )
    assert r7_corrupt.status_code == 400

    r7_large = client.post(
        "/api/analyze",
        files={"file": ("large.jpg", b"0" * (MAX_UPLOAD_BYTES + 100), "image/jpeg")},
    )
    assert r7_large.status_code == 413

    # Case 8: Image with no face and no usable skin region (solid blue image)
    blue_img = np.zeros((300, 300, 3), dtype=np.uint8)
    blue_img[:, :] = (255, 0, 0)
    _, blue_buf = cv2.imencode(".png", blue_img)
    r8 = client.post(
        "/api/analyze",
        files={"file": ("non_skin.png", blue_buf.tobytes(), "image/png")},
    )
    assert r8.status_code == 422
    assert r8.json()["error_code"] == "no_skin_detected"

    # Compute latency summary
    lat_mean = statistics.mean(latencies)
    lat_median = statistics.median(latencies)
    lat_min = min(latencies)
    lat_max = max(latencies)

    # Compute peak GPU VRAM
    if torch.cuda.is_available():
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        vram_str = f"{peak_vram_mb:.1f} MB"
    else:
        vram_str = "CPU only (0 MB GPU)"

    # Print clean benchmark summary table to stdout (visible with pytest -s)
    print("\n" + "=" * 90)
    print("                     DERMASCAN FACE AI — DEMO CASES SUMMARY")
    print("=" * 90)
    for i, res in enumerate(demo_results, start=1):
        print(f"Case {i}: {res['desc']}")
        print(f"  image_id:     {res['image_id']}")
        print(f"  path:         {res['path']}")
        print(f"  input_mode:   {res['input_mode']} ({res['num_patches']} patches)")
        print(f"  prediction:   {res['prediction']} (confidence: {res['confidence'] * 100:.2f}%)")
        print(f"  abstained:    {res['abstained']}")
        print(f"  top2:         {res['top2']}")
        print(f"  latency:      {res['latency_ms']:.1f} ms")
        if res["caution_note"]:
            print(f"  caution_note: {res['caution_note']}")
        if res["abstention_message"]:
            print(f"  abstention:   {res['abstention_message']}")
        print("-" * 90)

    print("Case 7: Invalid Input (tested 415 invalid_file_type, 400 unreadable_image, 413 file_too_large) -> VERIFIED")
    print("Case 8: Non-Skin / No-Face Input (tested 422 no_skin_detected with bilingual detail) -> VERIFIED")
    print("=" * 90)
    print("                     END-TO-END LATENCY & VRAM BENCHMARK")
    print("=" * 90)
    print(f"Latency Mean:   {lat_mean:.1f} ms")
    print(f"Latency Median: {lat_median:.1f} ms")
    print(f"Latency Min:    {lat_min:.1f} ms")
    print(f"Latency Max:    {lat_max:.1f} ms")
    print(f"Peak GPU VRAM:  {vram_str}")
    print("=" * 90 + "\n")
