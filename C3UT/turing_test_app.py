import json
import random
import re
import logging
from pathlib import Path
from datetime import datetime

import streamlit as st
from PIL import Image

# ---------------------------
# Basic Configuration
# ---------------------------
DATA_ROOT = Path("./tuning_test")
RESULT_DIR = Path("./result")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

MARKERS = ["CD3", "CD4", "CD8", "CD19", "CD68", "FOXP3"]
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

try:
    st.set_option("deprecation.showwarning", False)
except Exception:
    pass
logging.getLogger("streamlit").setLevel(logging.ERROR)

# Standard filename parsing: {marker}_{real|fake}_{if|he}_{NNNN}.ext
NAME_RE = re.compile(
    r"^(?P<marker>CD3|CD4|CD8|CD19|CD68|FOXP3)_(?P<src>real|fake)_(?P<typ>if|he)_(?P<idx>\d+)\.(png|jpg|jpeg|tif|tiff)$",
    re.IGNORECASE,
)

def parse_norm_name(p: Path):
    m = NAME_RE.match(p.name)
    if not m:
        return None
    g = m.groupdict()
    return {
        "marker": g["marker"].upper(),
        "source": g["src"].lower(),
        "type": g["typ"].lower(),
        "index": int(g["idx"]),
        "path": p,
    }

# ---------------------------
# Data Loading & Pairing
# ---------------------------
def list_if_he_pairs(marker: str):
    base = DATA_ROOT / marker
    real_he_dir = base / "real" / "he"
    real_if_dir = base / "real" / "if"
    fake_he_dir = base / "fake" / "he"
    fake_if_dir = base / "fake" / "if"

    def build_pairs(he_dir: Path, if_dir: Path, source_label: str):
        if not he_dir.exists() or not if_dir.exists():
            return []
        he_idx = {}
        for p in he_dir.iterdir():
            if p.is_file():
                meta = parse_norm_name(p)
                if meta and meta["marker"] == marker and meta["source"] == source_label and meta["type"] == "he":
                    he_idx[meta["index"]] = meta["path"]
        pairs = []
        for p in if_dir.iterdir():
            if p.is_file():
                meta = parse_norm_name(p)
                if meta and meta["marker"] == marker and meta["source"] == source_label and meta["type"] == "if":
                    he_p = he_idx.get(meta["index"])
                    if he_p:
                        pairs.append((he_p, meta["path"]))
        return sorted(pairs, key=lambda x: x[1].name)

    real_pairs = build_pairs(real_he_dir, real_if_dir, "real")
    fake_pairs = build_pairs(fake_he_dir, fake_if_dir, "fake")
    return real_pairs, fake_pairs

def pick_question_from_pool(pool, avoid_if_path=None):
    if not pool:
        return None
    if len(pool) == 1:
        return pool[0]
    candidates = [p for p in pool if str(p[1]) != str(avoid_if_path)]
    if not candidates:
        candidates = pool
    return random.choice(candidates)

# ---------------------------
# Result Reading/Writing
# ---------------------------
def append_or_update_result(doctor_name: str, record: dict):
    out_path = RESULT_DIR / f"{doctor_name}.json"
    try:
        data = json.loads(out_path.read_text("utf-8")) if out_path.exists() else []
        if not isinstance(data, list):
            data = []
    except Exception:
        data = []

    idx = None
    for i, r in enumerate(data):
        if isinstance(r, dict) and r.get("uuid") == record.get("uuid"):
            idx = i
            break
    if idx is None:
        data.append(record)
    else:
        data[idx] = record

    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def load_existing_records(doctor_name: str):
    out_path = RESULT_DIR / f"{doctor_name}.json"
    if not out_path.exists():
        return []
    try:
        data = json.loads(out_path.read_text("utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []

# ---------------------------
# Page & Session State
# ---------------------------
st.set_page_config(page_title="HE-IF Evaluation", layout="wide")
st.markdown(
"""
<style>
/* Layout & Container */
.block-container {
padding-top: 0.9rem; /* Increase top padding slightly */
padding-bottom: 0.6rem;
max-width: 1400px;
overflow: visible !important;
}

/* Allow overflow for header and app root */
header, .stApp header, [data-testid="stHeader"], .stApp, [data-testid="stAppViewContainer"] {
    overflow: visible !important;
}

/* Title styles to prevent clipping */
h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    margin-top: 0.4rem;
    margin-bottom: 0.6rem;
    line-height: 1.25;
    word-break: keep-all;
    overflow-wrap: normal;
    hyphens: none;
}

/* Limit image size */
[data-testid="stImage"] img {
    max-height: 65vh;
    object-fit: contain;
}

/* Form controls details */
.stRadio > div { gap: 0.4rem; }
.stButton > button { padding: 0.35rem 0.8rem; }
</style>
""",
unsafe_allow_html=True,
)

st.sidebar.header("Identity Confirmation")
doctor_name = st.sidebar.text_input("Doctor Name (for saving results)", value="", max_chars=64, placeholder="Enter name to start")
confirm = st.sidebar.button("Confirm & Start", type="primary", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.header("Settings")
marker = st.sidebar.selectbox("Marker", MARKERS, index=0, disabled=not doctor_name.strip())
no_replacement = st.sidebar.checkbox("Avoid Duplicates (No replacement in session)", value=True, disabled=not doctor_name.strip())

# Initialize State
if "questions_history" not in st.session_state:
    st.session_state.questions_history = []
if "current_idx" not in st.session_state:
    st.session_state.current_idx = -1
if "last_marker" not in st.session_state:
    st.session_state.last_marker = marker
if "used_pairs" not in st.session_state:
    st.session_state.used_pairs = {}
if "results_cache" not in st.session_state:
    st.session_state.results_cache = {}
if "done_if_paths_by_marker" not in st.session_state:
    st.session_state.done_if_paths_by_marker = {}
if "totals_by_marker" not in st.session_state:
    st.session_state.totals_by_marker = {}
if "last_doctor" not in st.session_state:
    st.session_state.last_doctor = ""
if "no_replacement_effective" not in st.session_state:
    st.session_state.no_replacement_effective = True

def get_used_set(marker: str, source: str):
    st.session_state.used_pairs.setdefault(marker, {})
    st.session_state.used_pairs[marker].setdefault(source, set())
    return st.session_state.used_pairs[marker][source]

def set_total_for_marker(marker: str):
    real_pairs, fake_pairs = list_if_he_pairs(marker)
    st.session_state.totals_by_marker[marker] = len(real_pairs) + len(fake_pairs)

def get_total_for_marker(marker: str):
    if marker not in st.session_state.totals_by_marker:
        set_total_for_marker(marker)
    return st.session_state.totals_by_marker[marker]

def refresh_done_from_disk(active_doctor: str):
    """Load existing records for the doctor to avoid duplicate labeling."""
    if not active_doctor:
        return
    records = load_existing_records(active_doctor)
    by_marker = {}
    for r in records:
        m = r.get("marker")
        ifp = r.get("if_path")
        if m and ifp:
            by_marker.setdefault(m, set()).add(ifp)
        u = r.get("uuid")
        if u:
            st.session_state.results_cache[u] = r
    st.session_state.done_if_paths_by_marker = by_marker

def reset_for_marker_change(new_marker: str):
    st.session_state.questions_history = []
    st.session_state.current_idx = -1
    st.session_state.last_marker = new_marker
    set_total_for_marker(new_marker)

def ensure_first_question():
    if st.session_state.current_idx == -1:
        return new_question()

def new_question():
    """Draw a new question and append to history; try to avoid duplicates."""
    marker_local = st.session_state.last_marker
    real_pairs, fake_pairs = list_if_he_pairs(marker_local)
    pools = []
    if real_pairs:
        pools.append(("real", real_pairs))
    if fake_pairs:
        pools.append(("fake", fake_pairs))
    if not pools:
        return None

    done_set = st.session_state.done_if_paths_by_marker.get(marker_local, set())
    last_if = None
    if st.session_state.current_idx >= 0:
        last_if = st.session_state.questions_history[st.session_state.current_idx]["if_path"]

    combined = []
    for src, pool in pools:
        if st.session_state.no_replacement_effective:
            used = get_used_set(marker_local, src)
            candidates = [p for p in pool if str(p[1]) not in used and str(p[1]) not in done_set]
        else:
            candidates = [p for p in pool if str(p[1]) not in done_set]
        if candidates:
            combined.append((src, candidates))

    if not combined:
        return None

    src, cand_pool = random.choice(combined) if len(combined) > 1 else combined[0]
    pair = pick_question_from_pool(cand_pool, avoid_if_path=last_if)
    if not pair:
        return None

    he_path, if_path = pair
    q = {
        "marker": marker_local,
        "source": src,
        "he_path": str(he_path),
        "if_path": str(if_path),
        "uuid": f"{marker_local}-{src}-{Path(if_path).stem}-{random.randint(10**5, 10**6-1)}",
    }

    if st.session_state.no_replacement_effective:
        get_used_set(marker_local, src).add(str(if_path))

    st.session_state.questions_history.append(q)
    st.session_state.current_idx = len(st.session_state.questions_history) - 1
    return q

# Handle Identity and State Sync
doctor_clean = doctor_name.strip()

if (confirm and doctor_clean) or (doctor_clean and st.session_state.last_doctor != doctor_clean):
    st.session_state.last_doctor = doctor_clean
    st.session_state.results_cache = {}
    st.session_state.used_pairs = {}
    st.session_state.questions_history = []
    st.session_state.current_idx = -1
    st.session_state.no_replacement_effective = bool(no_replacement)
    refresh_done_from_disk(doctor_clean)
    reset_for_marker_change(marker)

if doctor_clean and st.session_state.last_marker != marker:
    reset_for_marker_change(marker)

if doctor_clean:
    st.session_state.no_replacement_effective = bool(no_replacement)

if doctor_clean:
    total_count = get_total_for_marker(st.session_state.last_marker)
    done_count = len(st.session_state.done_if_paths_by_marker.get(st.session_state.last_marker, set()))
else:
    total_count = 0
    done_count = 0

# ---------------------------
# Page Content
# ---------------------------
st.title("HE-IF Evaluation")

if not doctor_clean:
    st.info("Please enter your name in the sidebar and click 'Confirm & Start'.")
    st.stop()

st.caption(f"Doctor: {doctor_clean} · Marker: {st.session_state.last_marker}")

# Ensure first question is generated after identity confirmation
ensure_first_question()

# Current Question
q = None
if 0 <= st.session_state.current_idx < len(st.session_state.questions_history):
    q = st.session_state.questions_history[st.session_state.current_idx]

# Session Stats: Current index and total drawn
hist_total = len(st.session_state.questions_history)
curr_no = st.session_state.current_idx + 1 if q else 0

if q is None:
    st.warning("No questions available. Please check data paths or switch Marker.")
else:
    col1, col2 = st.columns([1, 1])
    with col1:
        he_img = Image.open(q["he_path"]).convert("RGB")
        # Resize: max dimension 800
        if max(he_img.size) > 800:
            he_img.thumbnail((800, 800))
        st.image(he_img, caption="HE", use_container_width=True)
    with col2:
        if_img = Image.open(q["if_path"]).convert("RGB")
        if max(if_img.size) > 800:
            if_img.thumbnail((800, 800))
        st.image(if_img, caption="IF", use_container_width=True)

    cached = st.session_state.results_cache.get(q["uuid"], {})
    default_pred_idx = 0 if cached.get("doctor_pred", "real") == "real" else 1
    default_quality = int(cached.get("quality_score", 3))
    default_note = cached.get("note", "")

    st.markdown("---")
    with st.form("judge_form", clear_on_submit=False):
        pred = st.radio("Is the IF image Real?", options=["Real", "Fake"], index=default_pred_idx, horizontal=True)
        quality = st.select_slider("IF Staining Quality", options=[1, 2, 3, 4, 5], value=default_quality)
        note = st.text_input("Notes (Optional)", value=default_note)
        submitted = st.form_submit_button("Save Result")

    if submitted:
        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "doctor_name": doctor_clean,
            "marker": q["marker"],
            "he_path": q["he_path"],
            "if_path": q["if_path"],
            "true_source": q["source"],
            "doctor_pred": "real" if pred == "Real" else "fake",
            "quality_score": int(quality),
            "uuid": q["uuid"],
            "note": note,
        }
        append_or_update_result(doctor_clean, record)
        st.session_state.results_cache[q["uuid"]] = record
        st.session_state.done_if_paths_by_marker.setdefault(q["marker"], set()).add(q["if_path"])
        st.success("Saved successfully.")

        # Auto-advance
        advanced = False
        if st.session_state.current_idx < len(st.session_state.questions_history) - 1:
            st.session_state.current_idx += 1
            advanced = True
        else:
            created = new_question()
            if created is not None:
                advanced = True

        if advanced:
            st.rerun()
        else:
            st.warning("No more images available (Current marker finished or no unlabelled samples).")

    # Re-calculate for display
    done_count = len(st.session_state.done_if_paths_by_marker.get(st.session_state.last_marker, set()))
    total_count = get_total_for_marker(st.session_state.last_marker)

    # Navigation & Progress
    colA, colB, colC, colD, colE = st.columns([1, 1, 3, 4, 3])
    with colA:
        disabled_prev = st.session_state.current_idx <= 0
        if st.button("Previous", disabled=disabled_prev):
            if st.session_state.current_idx > 0:
                st.session_state.current_idx -= 1
                st.rerun()
    with colB:
        if st.button("Next"):
            moved = False
            if st.session_state.current_idx < len(st.session_state.questions_history) - 1:
                st.session_state.current_idx += 1
                moved = True
            else:
                created = new_question()
                if created is not None:
                    moved = True
            if moved:
                st.rerun()
            else:
                st.warning("No more images available (Current marker finished or no unlabelled samples).")
    with colC:
        st.progress(min(done_count, total_count) / max(total_count, 1))
    with colD:
        st.write(f"Labeled: {min(done_count, total_count)} / Total: {total_count}")
    with colE:
        st.write(f"Current: {curr_no} / Drawn: {hist_total}")

# Footer
st.caption("Results will be saved to ./result/{doctor_name}.json")