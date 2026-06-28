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
                         cache_dir):
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
        top_journals = rerank_journals(top_journals)
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
    if score >= 40: return "High Match"
    if score >= 25: return "Medium Match"
    return "Low Match"


def derive_confidence(score):
    if score >= 40: return "High"
    if score >= 25: return "Medium"
    return "Low"


def derive_why_recommended(j):
    m = j["coverage_metrics"]
    reasons = []
    if j["Aims_Scope_Sim"] >= 0.8:
        reasons.append(f"High aims & scope similarity ({j['Aims_Scope_Sim']:.2f})")
    if m["scientific_domains_aimscope"] >= 0.5:
        reasons.append("Scientific domain alignment with journal aims & scope")
    if m["research_focuses_coverage_aimscope"] >= 0.5:
        reasons.append("Research focus areas represented in journal scope")
    if m["scientific_domains_category_coverage"] > 0:
        reasons.append("Scientific domain category overlap detected")
    if m["research_focuses_category_coverage"] > 0:
        reasons.append("Research focus category overlap detected")
    if not reasons:
        reasons.append("General relevance based on semantic similarity")
    return reasons


def derive_journal_profile(j):
    sci_domains  = j["extracted_journal_features"]["sci_evi"]
    research_evi = j["extracted_journal_features"]["research_evi"]
    research_focuses = [{"name": rf, "status": "covered"} for rf in research_evi]
    return {"scientific_domains": sci_domains, "research_focuses": research_focuses}


def derive_feature_contribution(j):
    m = j["coverage_metrics"]
    return [
        {"feature": "Base Score (BioBERT)",  "weight": 0.25, "raw": min(j["Base_Score"], 1.0),                    "color": "#f97316"},
        {"feature": "Aims & Scope Sim.",     "weight": 0.20, "raw": j["Aims_Scope_Sim"],                           "color": "#2563eb"},
        {"feature": "Sci. Domain Coverage",  "weight": 0.20, "raw": m["scientific_domains_coverage"],              "color": "#14b8a6"},
        {"feature": "Sci. Domain Aims",      "weight": 0.15, "raw": m["scientific_domains_aimscope"],              "color": "#8b5cf6"},
        {"feature": "Domain Category Cov.",  "weight": 0.10, "raw": m["scientific_domains_category_coverage"],     "color": "#ec4899"},
        {"feature": "Research Focus Aims",   "weight": 0.10, "raw": m["research_focuses_coverage_aimscope"],       "color": "#6366f1"},
    ]


def derive_risk_analysis(j):
    m = j["coverage_metrics"]
    strengths, risks = [], []
    if j["Aims_Scope_Sim"] >= 0.8:
        strengths.append(f"High aims & scope similarity ({j['Aims_Scope_Sim']:.3f})")
    if m["scientific_domains_aimscope"] >= 0.5:
        strengths.append("Scientific domain alignment with journal aims")
    if m["research_focuses_coverage_aimscope"] >= 0.5:
        strengths.append("Research focus areas covered in journal scope")
    if j["Rerank"]["final_fit_score"] >= 30:
        strengths.append("Above-average overall fit score")
    if not strengths:
        strengths.append("Semantic relevance to paper topic")
    if m["scientific_domains_coverage"] == 0:
        risks.append("No direct scientific domain overlap detected")
    if not risks:
        risks.append("No significant weaknesses identified")
    return {"strengths": strengths, "risks": risks}


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
        j["Match_Level"]          = derive_match_level(j["Rerank"]["final_fit_score"])
        j["Confidence"]           = derive_confidence(j["Rerank"]["final_fit_score"])
        j["why_recommended"]      = derive_why_recommended(j)
        j["journal_profile"]      = derive_journal_profile(j)
        j["feature_contribution"] = derive_feature_contribution(j)
        j["risk_analysis"]        = derive_risk_analysis(j)
    return journals


def load_result_file():
    """Load outputs/result.json, enrich, and store in session state."""
    raw = json.loads(RESULT_FILE.read_text(encoding="utf-8"))
    paper    = prepare_paper(raw["paper_information"])
    journals = sorted(raw["Top10_journals"], key=lambda j: j["Rerank"]["new_rank"])
    journals = enrich_journals(journals)
    st.session_state["result"]       = {"paper": paper, "journals": journals}
    st.session_state["selected_idx"] = 0
    st.session_state["right_view"]   = "information"


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
for _k, _v in {"result": None, "selected_idx": 0, "right_view": "information",
                "show_all": False}.items():
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
                          value=r"D:\File\KLTN\Medprj\models\Epoch_02_SIMCPRS_dmis-lab_biobert-v1_1_CL.pth")
    mname = st.text_input("Base model", value="dmis-lab/biobert-v1.1")
    dpath = st.text_input("Data folder", value=r"D:\File\KLTN\Medprj\data")

    with st.expander("Advanced options"):
        enc_model  = st.text_input("SPECTER2 model",   value="allenai/specter2_base")
        qwen_model = st.text_input("Qwen model",       value="Qwen/Qwen3.5-2B")
        features   = st.selectbox("Features", ["TAK", "TA", "TK", "T"], index=0)
        max_len    = st.number_input("Max length", value=512, min_value=128, max_value=1024, step=64)
        use_aim    = st.checkbox("Use aim embeddings", value=True)
        use_cat    = st.checkbox("Use category text",  value=False)

    if st.button("Load Models", type="primary", use_container_width=True):
        if not ckpt or not dpath:
            st.error("Checkpoint path and data folder are required.")
        else:
            with st.spinner("Loading models… (may take several minutes)"):
                try:
                    st.session_state["_models"] = get_pipeline_models(
                        ckpt, mname, dpath, enc_model, qwen_model,
                        features, max_len, use_aim, use_cat,
                        str(OUTPUT_DIR / "cache"),
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
  .block-container{padding-top:1.4rem;padding-bottom:2rem;max-width:1280px;}

  button[kind="primary"]{background:#2563eb!important;border-color:#2563eb!important;border-radius:8px!important;}
  button[kind="primary"]:hover{background:#1d4ed8!important;}
  button[kind="secondary"]{border-radius:8px!important;}

  .section-title{font-size:1.05rem;font-weight:700;color:#111827;
                 display:flex;align-items:center;gap:.4rem;margin-bottom:.5rem;}
  .label{font-size:.73rem;font-weight:600;color:#6b7280;text-transform:uppercase;
         letter-spacing:.05em;margin-bottom:.2rem;}

  .badge{display:inline-flex;align-items:center;padding:.15rem .6rem;border-radius:999px;
         font-size:.7rem;font-weight:600;line-height:1.4;}
  .badge-high{background:#dcfce7;color:#16a34a;}
  .badge-med {background:#fef3c7;color:#d97706;}
  .badge-low {background:#fee2e2;color:#dc2626;}
  .badge-blue{background:#dbeafe;color:#2563eb;}
  .badge-gray{background:#f3f4f6;color:#374151;}

  .jcard{border:1px solid #e2e8f0;border-radius:14px;padding:.95rem 1.1rem;
         margin-bottom:.7rem;background:#fff;transition:box-shadow .15s,border-color .15s;}
  .jcard:hover{box-shadow:0 4px 14px rgba(0,0,0,.08);}
  .jcard.active{border:2px solid #2563eb;background:#eff6ff;}
  .rank-circle{width:28px;height:28px;border-radius:50%;color:#fff;font-weight:700;
               font-size:.8rem;display:inline-flex;align-items:center;
               justify-content:center;flex-shrink:0;}

  .fit-num{font-size:1.4rem;font-weight:800;color:#111827;line-height:1.1;}
  .fit-den{font-size:.82rem;color:#9ca3af;font-weight:500;}
  .stars  {color:#f59e0b;font-size:.9rem;letter-spacing:1px;}
  .muted  {color:#6b7280;font-size:.78rem;}

  .check-row{display:flex;align-items:flex-start;gap:.4rem;margin:.28rem 0;
             font-size:.88rem;color:#1f2937;line-height:1.45;}
  .check{color:#22c55e;font-weight:700;flex-shrink:0;}
  .warn {color:#d97706;font-weight:700;flex-shrink:0;}

  .bar-track{background:#e5e7eb;border-radius:999px;height:7px;overflow:hidden;flex:1;}
  .bar-fill {height:7px;border-radius:999px;}

  .strengths-box{background:#f0fdf4;border:1px solid #bbf7d0;border-radius:10px;padding:.8rem 1rem;}
  .risks-box   {background:#fff7ed;border:1px solid #fed7aa;border-radius:10px;
                padding:.8rem 1rem;margin-top:.7rem;}
  .box-title{font-weight:700;font-size:.88rem;margin-bottom:.35rem;}

  .feat-row{display:flex;align-items:center;gap:.5rem;margin:.38rem 0;}
  .feat-lbl{font-size:.8rem;color:#374151;flex:0 0 145px;white-space:nowrap;
            overflow:hidden;text-overflow:ellipsis;}
  .feat-wt {font-size:.75rem;color:#9ca3af;flex:0 0 28px;text-align:right;}
  .feat-val{font-size:.8rem;font-weight:700;flex:0 0 38px;text-align:right;}

  .pill-id{display:inline-block;padding:.1rem .45rem;border-radius:5px;
           background:#eef2ff;color:#4f46e5;font-size:.7rem;font-weight:700;margin-left:.35rem;}

  .cov-card{background:#fff;border:1.5px solid #e2e8f0;border-radius:14px;
            padding:1.1rem 1.25rem;min-height:220px;}
  .cov-card-title{font-weight:700;font-size:.95rem;color:#111827;margin-bottom:.75rem;
                  padding-bottom:.45rem;border-bottom:1px solid #f1f5f9;}
  .cov-bar-row{display:flex;align-items:center;gap:.55rem;margin:.3rem 0 .8rem;}
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


def pct(x):
    return f"{round(x * 100)}%"


def badge(text, cls="badge-gray"):
    return f"<span class='badge {cls}'>{text}</span>"


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
    "<div style='font-size:1.4rem;font-weight:700;color:#111827;line-height:1.6;padding:.1rem 0;'>Journal Recommendation System</div>"
    "</div>",
    unsafe_allow_html=True,
)
st.divider()

# --------------------------------------------------------------------------- #
# Paper Input form
# --------------------------------------------------------------------------- #
st.markdown("<div class='section-title'> Paper Input</div>", unsafe_allow_html=True)

# Pre-fill from last result if available
_pre_T, _pre_A, _pre_K = "", "", ""
if RESULT_FILE.exists():
    try:
        _prev = json.loads(RESULT_FILE.read_text(encoding="utf-8"))
        _inp  = _prev["paper_information"]["inputs"]
        _pre_T = _inp.get("T", "")
        _pre_A = _inp.get("A", "")
        _pre_K = ", ".join(_inp.get("K", []))
    except Exception:
        pass

with st.form("paper_form"):
    fc1, fc2 = st.columns([2, 3])
    with fc1:
        st.markdown("**Title**")
        title_in    = st.text_input("Title", value=_pre_T, placeholder="Enter paper title…",
                                    label_visibility="collapsed")
        st.markdown("**Keywords**")
        keywords_in = st.text_input("Keywords (comma-separated)", value=_pre_K,
                                    placeholder="e.g. deep learning, stroke, clinical prediction",
                                    label_visibility="collapsed")
    with fc2:
        st.markdown("**Abstract**")
        abstract_in = st.text_area("Abstract", value=_pre_A, height=130,
                                   placeholder="Enter abstract…",
                                   label_visibility="collapsed")
    models_ready = "_models" in st.session_state
    _, btn_col, _ = st.columns([1, 2, 1])
    with btn_col:
        submitted = st.form_submit_button(
            "Recommend Journals",
            type="primary",
            use_container_width=True,
            disabled=not models_ready,
            help="Load models from the sidebar first" if not models_ready else
                 "Run the pipeline and save results to outputs/result.json",
        )

if submitted:
    if not title_in.strip():
        st.warning("Please enter a paper title.")
    else:
        raw = run_and_save(title_in, abstract_in, keywords_in, st.session_state["_models"])
        paper    = prepare_paper(raw["paper_information"])
        journals = sorted(raw["Top10_journals"], key=lambda j: j["Rerank"]["new_rank"])
        journals = enrich_journals(journals)
        st.session_state["result"]       = {"paper": paper, "journals": journals}
        st.session_state["selected_idx"] = 0
        st.session_state["right_view"]   = "information"

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
# Top Journals  +  Detail panel
# --------------------------------------------------------------------------- #
left, right = st.columns([5, 6], gap="large")

with left:
    st.markdown("<div class='section-title' style='font-size:1.35rem;'>🏆 Top Recommended Journals</div>",
                unsafe_allow_html=True)
    show_all = st.session_state["show_all"]
    limit    = len(journals) if show_all else min(3, len(journals))

    for idx in range(limit):
        j     = journals[idx]
        sel   = (idx == st.session_state["selected_idx"])
        color = RANK_COLORS[idx % len(RANK_COLORS)]
        score = j["Rerank"]["final_fit_score"]
        match = j["Match_Level"]
        bc    = "badge-high" if match == "High Match" else ("badge-med" if match == "Medium Match" else "badge-low")

        st.markdown(
            f"<div class='jcard{'  active' if sel else ''}'>"
            f"<div style='display:flex;gap:.75rem;align-items:flex-start;'>"
            f"<div style='padding-top:.2rem;flex-shrink:0;'>"
            f"<span class='rank-circle' style='background:{color};'>{idx+1}</span></div>"
            f"<div style='flex:1;min-width:0;'>"
            f"<div style='font-weight:700;font-size:.92rem;color:#111827;margin-bottom:.15rem;"
            f"overflow:hidden;text-overflow:ellipsis;white-space:nowrap;'>{j['Name']}</div>"
            f"<div><span class='muted'>Fit score&nbsp;</span>"
            f"<span class='fit-num'>{int(round(score))}</span>"
            f"<span class='fit-den'>/100</span></div></div>"
            f"<div style='text-align:right;flex-shrink:0;padding-top:.1rem;'>"
            f"{badge(match, bc)}<br>"
            f"<span class='stars'>{stars_from_score(score)}</span><br>"
            f"<span class='muted'>Confidence: {j['Confidence']}</span></div>"
            f"</div></div>",
            unsafe_allow_html=True,
        )

        b1, b2 = st.columns(2)
        with b1:
            if st.button("📄 Info", key=f"info_{idx}", use_container_width=True):
                st.session_state["selected_idx"] = idx
                st.session_state["right_view"]   = "information"
                st.rerun()
        with b2:
            if st.button("💬 Explain", key=f"expl_{idx}", use_container_width=True):
                st.session_state["selected_idx"] = idx
                st.session_state["right_view"]   = "explanation"
                st.rerun()

    remaining = len(journals) - 3
    if remaining > 0:
        lbl = "Show fewer ▲" if show_all else f"Show {remaining} more ▾"
        if st.button(lbl, key="toggle_all", use_container_width=True):
            st.session_state["show_all"] = not show_all
            st.rerun()

with right:
    sj   = journals[st.session_state["selected_idx"]]
    name_short = sj["Name"][:44] + ("…" if len(sj["Name"]) > 44 else "")
    match = sj["Match_Level"]
    bc    = "badge-high" if match == "High Match" else ("badge-med" if match == "Medium Match" else "badge-low")

    st.markdown(
        f"<div style='margin-bottom:.55rem;'>"
        f"<div class='section-title'>{name_short}"
        f"<span class='pill-id'>#{sj['Rerank']['new_rank']}</span></div>"
        f"<div>{badge(match, bc)}&nbsp;"
        f"<span class='muted'>Score: {sj['Rerank']['final_fit_score']:.1f}/100"
        f"&nbsp;·&nbsp;Confidence: {sj['Confidence']}</span></div></div>",
        unsafe_allow_html=True,
    )

    tab_i, tab_e = st.tabs(["📄 Information", "💬 Explanation"])

    with tab_i:
        st.markdown("**Aims & Scope**")
        _aims_words = sj["Aims"].split()
        _aims_key   = f"aims_exp_{st.session_state['selected_idx']}"
        if _aims_key not in st.session_state:
            st.session_state[_aims_key] = False
        if len(_aims_words) <= 50 or st.session_state[_aims_key]:
            st.write(sj["Aims"])
            if len(_aims_words) > 50:
                if st.button("See less ▲", key=f"aims_btn_{st.session_state['selected_idx']}"):
                    st.session_state[_aims_key] = False
                    st.rerun()
        else:
            st.write(" ".join(_aims_words[:50]) + "…")
            if st.button("See more ▾", key=f"aims_btn_{st.session_state['selected_idx']}"):
                st.session_state[_aims_key] = True
                st.rerun()
        ci1, ci2 = st.columns(2)
        with ci1:
            st.markdown("**Categories (Scientific Domains)**")
            for d in sj["journal_profile"]["scientific_domains"]: st.markdown(f"- {d}")
        with ci2:
            st.markdown("**Research Focuses**")
            for rf in sj["journal_profile"]["research_focuses"]:
                #icon = "✅" if rf["status"] == "covered" else "⚠️"
                st.markdown(f"- {rf['name']}")
    with tab_e:
        expl = sj["Explanation"]
        st.markdown("**Main reasoning**");     st.info(expl["main_reasoning"])
        if expl.get("weakness_warning"):
            st.markdown("**Weakness warning**"); st.warning(expl["weakness_warning"])
        st.markdown("**Recommendation signals**")
        for w in sj["why_recommended"]:
            st.markdown(f"<div class='check-row'><span class='check'>✔</span>{w}</div>",
                        unsafe_allow_html=True)

st.divider()

# --------------------------------------------------------------------------- #
# Coverage Analysis
# --------------------------------------------------------------------------- #
sj = journals[st.session_state["selected_idx"]]
st.markdown("<div class='section-title'>📊 Coverage Analysis</div>", unsafe_allow_html=True)
cv1, cv2, cv3 = st.columns(3)

_RES_LIMIT = 5

with cv1:
    p_res     = paper["paper_profile"]["research_focuses"]
    _pk       = "cov_paper_res_exp"
    _p_expanded = st.session_state.get(_pk, False)
    rows_sci  = "".join(
        f"<div class='check-row'><span class='check'>✔</span>{d}</div>"
        for d in paper["paper_profile"]["scientific_domains"]
    )
    rows_res  = "".join(
        f"<div class='check-row'><span class='check'>✔</span>{r}</div>"
        for r in (p_res if _p_expanded or len(p_res) <= _RES_LIMIT else p_res[:_RES_LIMIT])
    )
    st.markdown(
        f"<div class='cov-card'>"
        f"<div class='cov-card-title'>Paper Profile</div>"
        f"<div class='label'>Scientific Domains</div>{rows_sci}"
        f"<div class='label' style='margin-top:.8rem;'>Research Focuses</div>{rows_res}"
        f"</div>",
        unsafe_allow_html=True,
    )
    if len(p_res) > _RES_LIMIT:
        _lbl = "Show fewer ▲" if _p_expanded else f"Show {len(p_res) - _RES_LIMIT} more ▾"
        if st.button(_lbl, key="btn_paper_res", use_container_width=True):
            st.session_state[_pk] = not _p_expanded
            st.rerun()

with cv2:
    j_res        = sj["journal_profile"]["research_focuses"]
    _jk          = f"cov_j_res_exp_{st.session_state['selected_idx']}"
    _j_expanded  = st.session_state.get(_jk, False)
    j_name_short = sj["Name"][:28] + ("…" if len(sj["Name"]) > 28 else "")
    rows_sci_j   = "".join(
        f"<div class='check-row'><span class='check'>✔</span>{d}</div>"
        for d in sj["journal_profile"]["scientific_domains"]
    )
    rows_res_j   = "".join(
        f"<div class='check-row'><span class='check'>✔</span>{rf['name']}</div>"
        for rf in (j_res if _j_expanded or len(j_res) <= _RES_LIMIT else j_res[:_RES_LIMIT])
    )
    st.markdown(
        f"<div class='cov-card'>"
        f"<div class='cov-card-title'>Journal Profile "
        f"<span style='font-weight:500;color:#6b7280;font-size:.8rem;'>({j_name_short})</span></div>"
        f"<div class='label'>Scientific Domains</div>{rows_sci_j}"
        f"<div class='label' style='margin-top:.8rem;'>Research Focuses</div>{rows_res_j}"
        f"</div>",
        unsafe_allow_html=True,
    )
    if len(j_res) > _RES_LIMIT:
        _lbl = "Show fewer ▲" if _j_expanded else f"Show {len(j_res) - _RES_LIMIT} more ▾"
        if st.button(_lbl, key=f"btn_j_res_{st.session_state['selected_idx']}", use_container_width=True):
            st.session_state[_jk] = not _j_expanded
            st.rerun()

with cv3:
    m       = sj["coverage_metrics"]
    aims_p  = round(min(sj["Aims_Scope_Sim"], 1.0) * 100)
    sci_p   = round(m["scientific_domains_coverage"] * 100)
    res_p   = round(m["research_focuses_category_coverage"] * 100)
    def _bar(pct_val, color):
        return (
            f"<div class='cov-bar-row'>"
            f"<div class='bar-track'>"
            f"<div class='bar-fill' style='background:{color};width:{pct_val}%;'></div>"
            f"</div>"
            f"<span style='font-weight:700;color:{color};font-size:.82rem;flex-shrink:0;'>"
            f"{pct_val}%</span></div>"
        )
    st.markdown(
        f"<div class='cov-card'>"
        f"<div class='cov-card-title'>Coverage Summary</div>"
        f"<div class='label'>Aims / Scope Similarity</div>{_bar(aims_p, '#2563eb')}"
        f"<div class='label'>Scientific Domain Coverage</div>{_bar(sci_p, '#2563eb')}"
        f"<div class='label'>Research Focus Coverage</div>{_bar(res_p, '#22c55e')}"
        f"</div>",
        unsafe_allow_html=True,
    )

st.divider()

# --------------------------------------------------------------------------- #
# Feature Contribution  +  Risk & Weakness  (2 columns, LTR block removed)
# --------------------------------------------------------------------------- #
sj = journals[st.session_state["selected_idx"]]
bot1, bot2 = st.columns([3, 2], gap="large")

with bot1:
    st.markdown("<div class='section-title'>📈 Feature Contribution</div>", unsafe_allow_html=True)
    st.caption("Each signal's weight × raw value → contribution to the final fit score")

    feats   = sj["feature_contribution"]
    max_c   = max(f["raw"] * f["weight"] for f in feats) or 1

    for f in feats:
        contrib = f["raw"] * f["weight"]
        width   = int(contrib / max_c * 100)
        st.markdown(
            f"<div class='feat-row'>"
            f"<span class='feat-lbl'>{f['feature']}</span>"
            f"<div class='bar-track'>"
            f"<div class='bar-fill' style='width:{width}%;background:{f['color']};'></div></div>"
            f"<span class='feat-wt' style='color:{f['color']};'>{int(f['weight']*100)}%</span>"
            f"<span class='feat-val' style='color:{f['color']};'>+{contrib*100:.1f}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.divider()
    mc1, mc2 = st.columns(2)
    mc1.metric("Final Fit Score",  f"{sj['Rerank']['final_fit_score']:.1f} / 100")
    mc2.metric("Rank Change",      f"#{sj['Rank']} → #{sj['Rerank']['new_rank']}")

with bot2:
    st.markdown("<div class='section-title'>🛡️ Risk & Weakness</div>", unsafe_allow_html=True)
    ra = sj["risk_analysis"]
    s_html = "".join(
        f"<div class='check-row'><span class='check'>✔</span>{s}</div>"
        for s in ra["strengths"]
    )
    st.markdown(
        f"<div class='strengths-box'>"
        f"<div class='box-title' style='color:#16a34a;'>✅ Strengths</div>{s_html}</div>",
        unsafe_allow_html=True,
    )
    r_html = "".join(
        f"<div class='check-row'><span class='warn'>⚠</span>{r}</div>"
        for r in ra["risks"]
    )
    st.markdown(
        f"<div class='risks-box'>"
        f"<div class='box-title' style='color:#d97706;'>⚠️ Potential Risks</div>{r_html}</div>",
        unsafe_allow_html=True,
    )
