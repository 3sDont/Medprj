"""
MedPRS – Explainable Journal Recommendation System

Flow:
  1. Configure model paths in the sidebar → "Load Models"
  2. Enter paper title / abstract / keywords
  3. Click "Analyze" → pipeline runs, result saved to outputs/result.json
  4. UI loads outputs/result.json and displays the recommendations

Run with:
    streamlit run app.py
"""

import os, sys, json, uuid
from pathlib import Path

import streamlit as st

_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT / "src"))

OUTPUT_DIR  = _ROOT / "outputs"
RESULT_FILE = OUTPUT_DIR / "result.json"

OUTPUT_DIR.mkdir(exist_ok=True)

# --------------------------------------------------------------------------- #
# Page config
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="MedPRS – Journal Recommender",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------- #
# Pipeline loader  (cached: models stay in memory across reruns)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def get_pipeline_models(checkpoint_path, model_name, data_path,
                         encoder_model, qwen_model,
                         features, max_len, use_aim, use_category,
                         cache_dir, ltr_model_path):
    from pipeline import load_pipeline
    return load_pipeline(
        checkpoint_path=checkpoint_path,
        model_name=model_name,
        data_path=data_path,
        encoder_model=encoder_model,
        qwen_model=qwen_model,
        features=features,
        max_len=int(max_len),
        use_aim=use_aim,
        use_category=use_category,
        cache_dir=cache_dir,
        ltr_model_path=ltr_model_path,
    )


def run_and_save(title, abstract, keywords, models):
    """
    Execute the 4-step pipeline with live progress, then write
    the result to outputs/result.json and return the result dict.
    """
    from inference import run_inference_single
    from aims_scope_sim import compute_aims_sim_single
    from llm_extract import process_llm_extraction
    from reasoning import rerank_journals, generate_all_explanations

    def _to_grouped(items):
        if not items:
            return {}
        if len(items) <= 3:
            return {"1": items}
        mid = (len(items) + 1) // 2
        return {"1": items[:mid], "2": items[mid:]} if items[mid:] else {"1": items[:mid]}

    paper_id = f"req_{uuid.uuid4().hex[:8]}"

    with st.status("Running analysis pipeline…", expanded=True) as status:

        st.write("**Step 1 / 4** — Classifier inference (BioBERT)")
        top_journals = run_inference_single(
            title=title, abstract=abstract, keywords=keywords,
            model=models.classifier, tokenizer=models.tokenizer,
            aims_embeddings=models.aims_embeddings, journal_df=models.journal_df,
            device=models.device, features=models.features,
            max_len=models.max_len, topk=10, use_aim=models.use_aim,
        )
        for j in top_journals:
            raw = str(models.journal_df.iloc[j["journal_idx"]].get("Categories", ""))
            j["Categories"] = [c.strip() for c in raw.split(",") if c.strip()]
            j["Best_Quartile"] = str(models.journal_df.iloc[j["journal_idx"]].get("Best Quartile", "") or "N/A")
        st.write("✅ Top-10 journals retrieved")

        st.write("**Step 2 / 4** — Aims/scope similarity (SPECTER2)")
        top_journals = compute_aims_sim_single(
            title=title, abstract=abstract, keywords=keywords,
            top_journals=top_journals, encoder=models.specter2,
            aims_embs=models.specter_aims_embs, features=models.features,
        )
        st.write("✅ Aims_Scope_Sim computed")

        st.write("**Step 3 / 4** — Feature extraction & coverage (Qwen)")
        paper_features, top_journals = process_llm_extraction(
            title=title, abstract=abstract, keywords=keywords,
            top_journals=top_journals,
            journal_extracts=models.journal_extracts,
            extractor=models.qwen_extractor,
            encoder=models.specter2,
        )
        st.write("✅ Coverage metrics computed")

        st.write("**Step 4 / 4** — Re-ranking & explanations (Qwen)")
        top_journals = rerank_journals(top_journals, models.ltr_model)
        paper_info = {
            "title": title, "abstract": abstract, "keywords": keywords,
            "extracted_features": paper_features,
        }
        top_journals = generate_all_explanations(
            paper_info=paper_info, top_journals=top_journals,
            extractor=models.qwen_extractor, top_n=10,
        )
        for j in top_journals:
            j.pop("journal_idx", None)
            j.get("Rerank", {}).pop("rank_change", None)
        st.write("✅ Rankings ready")
        status.update(label="Analysis complete!", state="complete")

    kw_list = [k.strip() for k in keywords.split(",")] if keywords else []
    result = {
        "paper_id": paper_id,
        "paper_information": {
            "inputs": {"T": title, "A": abstract, "K": kw_list},
            "extracted_paper_features": {
                "sci_evidence":      _to_grouped(paper_features.get("scientific_domains", [])),
                "research_evidence": _to_grouped(paper_features.get("research_focuses", [])),
            },
        },
        "Top10_journals": top_journals,
    }

    RESULT_FILE.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    st.success(f"Result saved → `{RESULT_FILE}`")
    return result


# --------------------------------------------------------------------------- #
# Derivation helpers  (enrich raw JSON fields for the UI)
# --------------------------------------------------------------------------- #
def derive_match_level(score):
    if score >= 75: return "High Match"
    if score >= 50: return "Medium Match"
    return "Low Match"


def prepare_paper(paper):
    sci_domains, research_focuses = [], []
    for vals in paper["extracted_paper_features"]["sci_evidence"].values():
        sci_domains.extend(vals)
    for vals in paper["extracted_paper_features"]["research_evidence"].values():
        research_focuses.extend(vals)
    paper["paper_profile"] = {
        "scientific_domains": list(dict.fromkeys(sci_domains)),
        "research_focuses":   list(dict.fromkeys(research_focuses)),
    }
    return paper


def enrich_journals(journals):
    for j in journals:
        j["Match_Level"] = derive_match_level(j["Rerank"]["final_fit_score"])
    return journals


def load_result_file():
    """Load outputs/result.json, enrich, and store in session state."""
    raw = json.loads(RESULT_FILE.read_text(encoding="utf-8"))
    paper    = prepare_paper(raw["paper_information"])
    journals = sorted(raw["Top10_journals"], key=lambda j: j["Rerank"]["new_rank"])
    journals = enrich_journals(journals)
    st.session_state["result"]       = {"paper": paper, "journals": journals}
    st.session_state["selected_idx"] = 0


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
for _k, _v in {"result": None, "selected_idx": 0, "kw_chips": None}.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# --------------------------------------------------------------------------- #
# Sidebar – model configuration
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("## 🛡️ MedPRS")
    st.caption("Journal Recommendation")
    st.divider()

    st.markdown("### ⚙️ Model Configuration")
    ckpt  = st.text_input("Checkpoint (.pth)",
                          value=str(_ROOT / "models" / "Epoch_02_SIMCPRS_dmis-lab_biobert-v1_1_CL.pth"))
    mname = st.text_input("Base model", value="dmis-lab/biobert-v1.1")
    dpath = st.text_input("Data folder", value=str(_ROOT / "data"))

    with st.expander("Advanced options"):
        enc_model  = st.text_input("SPECTER2 model",   value="allenai/specter2_base")
        qwen_model = st.text_input("Qwen model",       value="Qwen/Qwen3.5-2B")
        features   = st.selectbox("Features", ["TAK", "TA", "TK", "T"], index=0)
        max_len    = st.number_input("Max length", value=512, min_value=128, max_value=1024, step=64)
        use_aim    = st.checkbox("Use aim embeddings", value=True)
        use_cat    = st.checkbox("Use category text",  value=False)
        ltr_path   = st.text_input("LTR model (fit score)",
                                    value=str(_ROOT / "models" / "student_model.json"),
                                    help="Trained model that turns Base_Score + coverage "
                                         "signals into the Fit Score. Leave blank or point "
                                         "to a missing file to fall back to raw Base_Score.")

    if st.button("Load Models", type="primary", use_container_width=True):
        if not ckpt or not dpath:
            st.error("Checkpoint path and data folder are required.")
        else:
            with st.spinner("Loading models… (may take several minutes)"):
                try:
                    st.session_state["_models"] = get_pipeline_models(
                        ckpt, mname, dpath, enc_model, qwen_model,
                        features, max_len, use_aim, use_cat,
                        str(OUTPUT_DIR / "cache"), ltr_path,
                    )
                    st.success("✅ Models loaded")
                except Exception as e:
                    st.error(f"Failed: {e}")

    if "_models" in st.session_state:
        st.success("✅ Pipeline ready")
    else:
        st.info("Load models to enable analysis")

    st.divider()
    st.markdown("### 🔬 Pipeline Steps")
    for num, col, name, desc in [
        ("1", "#f97316", "BioBERT Classifier",  "Top-10 candidate journals"),
        ("2", "#2563eb", "SPECTER2 Similarity", "Aims & scope alignment"),
        ("3", "#8b5cf6", "Qwen Extraction",     "Features + coverage metrics"),
        ("4", "#14b8a6", "Re-ranking",          "Fit scoring + explanations"),
    ]:
        st.markdown(
            f"<div style='display:flex;gap:.5rem;align-items:flex-start;margin:.4rem 0;'>"
            f"<span style='background:{col};color:#fff;border-radius:6px;padding:1px 8px;"
            f"font-weight:700;font-size:.78rem;flex-shrink:0;'>{num}</span>"
            f"<div><div style='font-weight:600;font-size:.82rem;color:#111;'>{name}</div>"
            f"<div style='font-size:.73rem;color:#6b7280;'>{desc}</div></div></div>",
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown("### 📁 Output")
    st.caption(f"`{RESULT_FILE}`")
    if RESULT_FILE.exists():
        if st.button("Load last result", use_container_width=True):
            load_result_file()
            st.rerun()
        st.caption("← reload without re-running")
    else:
        st.caption("No result file yet")

# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
st.markdown("""
<style>
  .stApp,[data-testid="stAppViewContainer"],[data-testid="stHeader"],
  [data-testid="stMain"],body{background:#f8fafc !important;}
  .block-container{padding-top:1.4rem;padding-bottom:2rem;max-width:1600px;}

  button[kind="primary"]{background:#2563eb!important;border-color:#2563eb!important;border-radius:8px!important;}
  button[kind="primary"]:hover{background:#1d4ed8!important;}
  button[kind="secondary"]{border-radius:8px!important;}

  .section-title{font-size:1.05rem;font-weight:700;color:#111827;
                 display:flex;align-items:center;gap:.4rem;margin-bottom:.5rem;}
  .label{font-size:.73rem;font-weight:600;color:#6b7280;text-transform:uppercase;
         letter-spacing:.05em;margin-bottom:.2rem;}

  .badge{display:inline-flex;align-items:center;justify-content:center;width:118px;
         padding:.25rem .8rem;border-radius:999px;
         font-size:.95rem;font-weight:600;line-height:1.4;white-space:nowrap;}
  .badge-high{background:#dcfce7;color:#16a34a;}
  .badge-med {background:#fef3c7;color:#d97706;}
  .badge-low {background:#fee2e2;color:#dc2626;}
  .badge-blue{background:#dbeafe;color:#2563eb;}
  .badge-gray{background:#f3f4f6;color:#374151;}

  .rank-circle{width:34px;height:34px;border-radius:50%;color:#fff;font-weight:700;
               font-size:.95rem;display:inline-flex;align-items:center;
               justify-content:center;flex-shrink:0;}

  .fit-num{font-size:1.3rem;font-weight:800;color:#111827;line-height:1.1;}
  .fit-den{font-size:.78rem;color:#9ca3af;font-weight:500;}
  .stars  {color:#f59e0b;font-size:1.35rem;letter-spacing:1px;}
  .muted  {color:#6b7280;font-size:.78rem;}

  .check-row{display:flex;align-items:flex-start;gap:.4rem;margin:.28rem 0;
             font-size:.88rem;color:#1f2937;line-height:1.45;}
  .check{color:#22c55e;font-weight:700;flex-shrink:0;}
  .warn {color:#d97706;font-weight:700;flex-shrink:0;}

  .bar-track{background:#e5e7eb;border-radius:999px;height:7px;overflow:hidden;flex:1;}
  .bar-fill {height:7px;border-radius:999px;}

  .cov-card-title{font-weight:700;font-size:.95rem;color:#111827;margin-bottom:.75rem;
                  padding-bottom:.45rem;border-bottom:1px solid #f1f5f9;}

  .reasoning-box{background:#fff;border:1px solid #e2e8f0;border-radius:10px;
                 padding:.9rem 1.1rem;font-size:.88rem;color:#1e3a5f;line-height:1.55;}
  .reasoning-title{font-weight:700;font-size:1.2rem;color:#111827;margin-bottom:.5rem;
                    display:flex;align-items:center;gap:.4rem;}

  div[data-testid="stButton"] > button[kind="secondary"]{
    background:#eff6ff;border:1px solid #bfdbfe;color:#1d4ed8;font-size:.75rem;
    padding:.15rem .5rem;
  }

  div[class*="st-key-select_"] button{
    background:linear-gradient(135deg,#7c3aed,#4f46e5) !important;border:1.5px solid transparent !important;
    color:#fff !important;border-radius:999px !important;font-weight:600 !important;
    box-shadow:0 2px 6px rgba(79,70,229,.35) !important;transition:all .15s ease !important;
  }
  div[class*="st-key-select_"] button:hover{
    box-shadow:0 4px 12px rgba(79,70,229,.5) !important;transform:translateY(-1px);
  }

  div.st-key-pi_title{background:#fff !important;min-height:90px;}
  div.st-key-pi_abstract, div.st-key-pi_keywords{background:#fff !important;min-height:210px;}

  div.st-key-paper_profile_card .cov-card-title{font-size:1.2rem;}
  div.st-key-paper_profile_card .label{font-size:.9rem;}
  div.st-key-paper_profile_card .check-row{font-size:1rem;}

  div[class*="st-key-jrow_"]{padding:.9rem 1.1rem !important;}

  div.st-key-detail_card .label{font-size:1.05rem;font-weight:700;color:#111827;}
  div.st-key-detail_card .check-row{font-size:.85rem;}

  div.st-key-detail_aims, div.st-key-detail_cats, div.st-key-detail_scores{
    background:#fff !important;
  }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
RANK_COLORS = ["#2563eb", "#14b8a6", "#8b5cf6", "#6366f1", "#ec4899",
               "#f97316", "#0ea5e9", "#84cc16", "#f59e0b", "#10b981"]


def stars_from_score(score):
    rating = round((score / 100) * 5 * 2) / 2
    full   = int(rating)
    half   = 1 if rating - full >= 0.5 else 0
    empty  = 5 - full - half
    return "★" * full + ("⯨" if half else "") + "☆" * empty


def badge(text, cls="badge-gray"):
    return f"<span class='badge {cls}'>{text}</span>"


def render_checklist(items, session_key, limit=5):
    """Checkmark list with a Show more/fewer toggle past `limit` items."""
    expanded = st.session_state.get(session_key, False)
    shown = items if (expanded or len(items) <= limit) else items[:limit]
    rows = "".join(f"<div class='check-row'><span class='check'>✔️</span>{it}</div>" for it in shown)
    st.markdown(rows or "<span class='muted'>None detected</span>", unsafe_allow_html=True)
    if len(items) > limit:
        lbl = "Show fewer ▲" if expanded else f"Show {len(items) - limit} more ▾"
        if st.button(lbl, key=f"btn_{session_key}", use_container_width=True):
            st.session_state[session_key] = not expanded
            st.rerun()


# The 4 real score_breakdown metrics (from reasoning.py's _EXPLAIN_SCORES)
# shown in the "Score Explanation" panel, with short display labels. Each
# row's percentage and description text are the real, already-computed
# metric and the LLM's own per-metric explanation — nothing is invented.
_SCORE_ITEMS = [
    ("base_score",       "Base Score",         "🧮"),
    ("aims_scope",       "Aims & Scope Match", "🎯"),
    ("domain_category",  "Domain Match",       "📖"),
    ("abstract_category","Abstract Match",     "📝"),
]
_SCORE_COLORS = {
    "base_score":        "#f97316",
    "aims_scope":        "#2563eb",
    "domain_category":   "#14b8a6",
    "abstract_category": "#ec4899",
}


def select_score_explanation(sj):
    by_key = {s["key"]: s for s in sj["Explanation"].get("score_breakdown", [])}
    items = []
    for key, label, icon in _SCORE_ITEMS:
        s = by_key.get(key, {})
        items.append({
            "key": key, "label": label, "icon": icon,
            "pct": s.get("value_pct", 0),
            "desc": s.get("explanation", ""),
        })
    return items


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.markdown(
    "<div style='display:flex;align-items:center;gap:1rem;padding:.6rem 0;overflow:visible;'>"
    "<span style='font-size:2.2rem;flex-shrink:0;line-height:1.8;'>🛡️</span>"
    "<div style='border-right:2px solid #e2e8f0;padding-right:1.1rem;flex-shrink:0;'>"
    "<div style='font-size:1.2rem;font-weight:900;color:#2563eb;letter-spacing:.02em;line-height:1.6;padding:.1rem 0;'>MedPRS</div>"
    "<div style='font-size:.72rem;color:#6b7280;font-weight:400;line-height:1.5;'>Research Assistant</div>"
    "</div>"
    "<div style='font-size:1.9rem;font-weight:700;color:#111827;line-height:1.6;padding:.1rem 0;'>Medical Journal Submission Recommendation System</div>"
    "</div>",
    unsafe_allow_html=True,
)
st.divider()

# --------------------------------------------------------------------------- #
# Paper Input form
# --------------------------------------------------------------------------- #
# Pre-fill from last result if available (only on first load of these keys)
if "title_in" not in st.session_state:
    _pre_T, _pre_A, _pre_K = "", "", []
    if RESULT_FILE.exists():
        try:
            _prev = json.loads(RESULT_FILE.read_text(encoding="utf-8"))
            _inp  = _prev["paper_information"]["inputs"]
            _pre_T = _inp.get("T", "")
            _pre_A = _inp.get("A", "")
            _pre_K = list(_inp.get("K", []))
        except Exception:
            pass
    st.session_state["title_in"]    = _pre_T
    st.session_state["abstract_in"] = _pre_A
    st.session_state["kw_chips"]    = _pre_K

if st.session_state["kw_chips"] is None:
    st.session_state["kw_chips"] = []


def _add_keyword():
    val = st.session_state.get("kw_add_input", "").strip()
    if val and val not in st.session_state["kw_chips"]:
        st.session_state["kw_chips"].append(val)
    st.session_state["kw_add_input"] = ""


models_ready = "_models" in st.session_state

st.markdown("<div class='section-title' style='font-size:1.55rem;'>📘 Paper Input</div>", unsafe_allow_html=True)

with st.container(border=True, key="pi_title"):
    st.markdown("**Title**")
    st.text_input("Title", key="title_in", placeholder="Enter paper title…",
                  label_visibility="collapsed")

fc2, fc3 = st.columns([3, 2.2], gap="medium")
with fc2:
    with st.container(border=True, key="pi_abstract"):
        st.markdown("**Abstract**")
        st.text_area("Abstract", key="abstract_in", height=130,
                     placeholder="Enter abstract…", label_visibility="collapsed")
with fc3:
    with st.container(border=True, key="pi_keywords"):
        st.markdown("**Keywords**")
        chips = st.session_state["kw_chips"]
        for i in range(0, len(chips), 2):
            row = chips[i:i + 2]
            for c, kw in zip(st.columns(len(row)), row):
                with c:
                    if st.button(f"{kw}  ✕", key=f"kwdel_{kw}", use_container_width=True):
                        st.session_state["kw_chips"].remove(kw)
                        st.rerun()
        st.text_input("Add keyword", key="kw_add_input",
                      placeholder="Add keyword and press Enter…",
                      label_visibility="collapsed", on_change=_add_keyword)

_, btn_col = st.columns([4, 1.3])
with btn_col:
    submitted = st.button(
        "⭐ Recommend Journals",
        key="recommend_btn",
        type="primary",
        use_container_width=True,
        disabled=not models_ready,
        help="Load models from the sidebar first" if not models_ready else
             "Run the pipeline and save results to outputs/result.json",
    )

if submitted:
    title_in    = st.session_state["title_in"]
    abstract_in = st.session_state["abstract_in"]
    keywords_in = ", ".join(st.session_state["kw_chips"])
    if "_models" not in st.session_state:
        st.warning("Please load models from the sidebar first.")
    elif not title_in.strip():
        st.warning("Please enter a paper title.")
    else:
        raw = run_and_save(title_in, abstract_in, keywords_in, st.session_state["_models"])
        paper    = prepare_paper(raw["paper_information"])
        journals = sorted(raw["Top10_journals"], key=lambda j: j["Rerank"]["new_rank"])
        journals = enrich_journals(journals)
        st.session_state["result"]       = {"paper": paper, "journals": journals}
        st.session_state["selected_idx"] = 0

# --------------------------------------------------------------------------- #
# Results gate
# --------------------------------------------------------------------------- #
if st.session_state["result"] is None:
    st.divider()
    if RESULT_FILE.exists():
        st.info("Use **Load last result** in the sidebar to view the previous analysis, "
                "or run a new analysis above.", icon="💡")
    else:
        st.info("Configure models in the sidebar, then submit a paper to see recommendations.",
                icon="💡")
    st.stop()

paper    = st.session_state["result"]["paper"]
journals = st.session_state["result"]["journals"]

st.divider()

# --------------------------------------------------------------------------- #
# Paper Profile (persistent left panel)  +  Journals & Detail (right)
# --------------------------------------------------------------------------- #
left, right = st.columns([1.6, 7.4], gap="large")

with left:
    with st.container(border=True, key="paper_profile_card"):
        st.markdown("<div class='cov-card-title'>👤 Paper Profile</div>", unsafe_allow_html=True)
        st.markdown("<div class='label'>Scientific Domains</div>", unsafe_allow_html=True)
        render_checklist(paper["paper_profile"]["scientific_domains"], "pp_sci_exp", limit=8)
        st.markdown("<hr style='margin:1rem 0 .8rem;border-color:#e2e8f0;'>", unsafe_allow_html=True)
        st.markdown("<div class='label'>Research Focuses</div>", unsafe_allow_html=True)
        render_checklist(paper["paper_profile"]["research_focuses"], "pp_res_exp", limit=5)

with right:
    st.markdown("<div class='section-title' style='font-size:1.55rem;'>🏆 Top Recommended Journals</div>",
                unsafe_allow_html=True)

    with st.container(height=460):
        for idx in range(len(journals)):
            j     = journals[idx]
            sel   = (idx == st.session_state["selected_idx"])
            color = RANK_COLORS[idx % len(RANK_COLORS)]
            score = j["Rerank"]["final_fit_score"]
            match = j["Match_Level"]
            bc    = "badge-high" if match == "High Match" else ("badge-med" if match == "Medium Match" else "badge-low")

            if sel:
                st.markdown(
                    f"<style>div.st-key-jrow_{idx}{{border:2px solid {color} !important;"
                    f"background:{color}0D !important;border-radius:12px !important;}}</style>",
                    unsafe_allow_html=True,
                )

            with st.container(border=sel, key=f"jrow_{idx}"):
                row_c = st.columns([0.5, 3.1, 1.6, 1.3, 1.6, 0.9])
                with row_c[0]:
                    st.markdown(f"<span class='rank-circle' style='background:{color};margin-top:.5rem;'>{idx+1}</span>",
                                unsafe_allow_html=True)
                with row_c[1]:
                    st.markdown(
                        f"<div style='font-weight:700;font-size:1.15rem;color:#111827;overflow:hidden;"
                        f"text-overflow:ellipsis;white-space:nowrap;'>{j['Name']}</div>"
                        f"<div><span class='muted'>Fit Score:&nbsp;</span>"
                        f"<span class='fit-num'>{int(round(score))}</span>"
                        f"<span class='fit-den'>/100</span></div>",
                        unsafe_allow_html=True,
                    )
                with row_c[2]:
                    st.markdown(f"<div style='margin-top:.6rem;'>{badge(match, bc)}</div>", unsafe_allow_html=True)
                with row_c[3]:
                    st.markdown(f"<div class='stars' style='margin-top:.55rem;'>{stars_from_score(score)}</div>",
                                unsafe_allow_html=True)
                with row_c[4]:
                    st.markdown(f"<div style='margin-top:.65rem;font-size:1.05rem;color:#6b7280;"
                                f"white-space:nowrap;'>Best Quartile: {j.get('Best_Quartile', 'N/A')}</div>",
                                unsafe_allow_html=True)
                with row_c[5]:
                    if st.button("View →", key=f"select_{idx}", use_container_width=True):
                        st.session_state["selected_idx"] = idx
                        st.rerun()
            st.markdown("<hr style='margin:.3rem 0;border-color:#f1f5f9;'>", unsafe_allow_html=True)

    st.markdown("<div style='height:1.4rem;'></div>", unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# Selected Journal Detail  (full page width, spans under Paper Profile too)
# --------------------------------------------------------------------------- #
sj    = journals[st.session_state["selected_idx"]]
score = sj["Rerank"]["final_fit_score"]
match = sj["Match_Level"]
bc    = "badge-high" if match == "High Match" else ("badge-med" if match == "Medium Match" else "badge-low")
color = RANK_COLORS[st.session_state["selected_idx"] % len(RANK_COLORS)]

st.markdown(f"<div class='section-title'>Selected Journal Detail — {sj['Name']}</div>",
            unsafe_allow_html=True)
st.markdown(
    f"<style>div.st-key-detail_card{{border:2px solid {color} !important;"
    f"box-shadow:0 4px 14px {color}22 !important;border-radius:14px !important;}}</style>",
    unsafe_allow_html=True,
)

with st.container(border=True, key="detail_card"):
    th1, th2 = st.columns([0.6, 8])
    with th1:
        st.markdown(f"<span class='rank-circle' style='background:{color};width:44px;height:44px;"
                    f"font-size:1.1rem;margin-top:.1rem;'>{sj['Rerank']['new_rank']}</span>",
                    unsafe_allow_html=True)
    with th2:
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:.7rem;'>"
            f"<span style='font-weight:700;font-size:1.3rem;color:#111827;'>{sj['Name']}</span>"
            f"{badge(match, bc)}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:.7rem;margin-top:.4rem;'>"
            f"<span class='muted' style='font-size:1rem;'>Overall Score:&nbsp;</span>"
            f"<span class='fit-num' style='font-size:1.5rem;'>{score:.1f}</span>"
            f"<span class='fit-den'>/100</span>"
            f"<span style='color:#d1d5db;font-size:1.3rem;'>|</span>"
            f"<span class='stars' style='font-size:1.6rem;'>{stars_from_score(score)}</span>"
            f"<span style='font-size:1.05rem;color:#6b7280;'>Best Quartile: "
            f"<b style='color:#111827;'>{sj.get('Best_Quartile', 'N/A')}</b></span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<hr style='margin:1.1rem 0;border-color:#e2e8f0;'>", unsafe_allow_html=True)

    left_col, right_col = st.columns([2, 1.3], gap="medium")

    with left_col:
        d1, d2 = st.columns(2, gap="medium")
        with d1, st.container(border=True, key="detail_aims", height=300):
            st.markdown("<div class='label'>🔬 Aims & Scope</div>", unsafe_allow_html=True)
            _aims_words = sj["Aims"].split()
            _aims_key   = f"aims_exp_{st.session_state['selected_idx']}"
            if len(_aims_words) <= 50 or st.session_state.get(_aims_key, False):
                st.markdown(f"<span style='font-size:.85rem;color:#374151;'>{sj['Aims']}</span>",
                            unsafe_allow_html=True)
                if len(_aims_words) > 50:
                    if st.button("See less ▲", key=f"aims_btn_{st.session_state['selected_idx']}"):
                        st.session_state[_aims_key] = False
                        st.rerun()
            else:
                st.markdown(f"<span style='font-size:.85rem;color:#374151;'>"
                            f"{' '.join(_aims_words[:50])}…</span>", unsafe_allow_html=True)
                if st.button("See more ▾", key=f"aims_btn_{st.session_state['selected_idx']}"):
                    st.session_state[_aims_key] = True
                    st.rerun()
        with d2, st.container(border=True, key="detail_cats", height=300):
            st.markdown("<div class='label'>🏷️ Categories </div>", unsafe_allow_html=True)
            render_checklist(sj.get("Categories", []), f"det_cats_{st.session_state['selected_idx']}", limit=5)

        st.markdown(
            f"<div class='reasoning-box' style='margin-top:1rem;height:168px;"
            f"overflow-y:auto;box-sizing:border-box;'>"
            f"<div class='reasoning-title'>🧠 Main Reasoning</div>"
            f"{sj['Explanation'].get('header', '')}</div>",
            unsafe_allow_html=True,
        )

    with right_col, st.container(border=True, key="detail_scores", height=500):
        st.markdown("<div class='label'>🌟 Score Explanation</div>", unsafe_allow_html=True)
        for g in select_score_explanation(sj):
            gcolor = _SCORE_COLORS[g["key"]]
            icon   = g["icon"]
            st.markdown(
                f"<div style='margin:1rem 0;'>"
                f"<div style='display:flex;justify-content:space-between;font-size:.95rem;'>"
                f"<span style='font-weight:600;color:#111827;'>{icon} {g['label']}</span>"
                f"<span style='font-weight:700;color:{gcolor};'>{g['pct']}%</span></div>"
                f"<div class='bar-track' style='margin-top:.4rem;height:10px;'>"
                f"<div class='bar-fill' style='width:{max(min(g['pct'],100),0)}%;background:{gcolor};height:10px;'></div></div>"
                f"<div style='margin-top:.3rem;font-size:.85rem;color:#6b7280;'>{g['desc']}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
