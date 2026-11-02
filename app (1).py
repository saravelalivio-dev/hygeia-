"""
HYGEIA - Nurturing the Breakthrough

Designing new antibiotics for drug-resistant Klebsiella pneumoniae.
Alivio, S. B. Y. | Cuering, P. F. T. | Oros, L. B. L.
Davao City National High School, 2026

Run with:  streamlit run app.py

    hygeia_data.py       all the numbers from the paper
    hygeia_features.py   the physicochemical equations
    hygeia_models.py     screening model, SHAP, Pareto sort
    hygeia_structure.py  3-D viewer for docked complexes
    hygeia_theme.py      colours, CSS, page layout
    app.py               this file: layout and wording

Plain wording in the headings and captions, the formal term underneath, and
the equations inside the "show the maths" boxes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

import hygeia_features as hf
import hygeia_models as hm
import hygeia_structure as hs
import hygeia_theme as theme
from hygeia_data import (
    ACPE_THRESHOLDS,
    CANDIDATES,
    CLASSIFIER_METRICS,
    CONFUSION,
    DOCKING_POWER,
    DOCKING_TARGETS,
    FEATURE_ABLATION,
    FINAL_LEADS,
    MAX_NEGATIVE_PHI,
    NMA_COMPLEXES,
    PIPELINE_STAGES,
    PROJECT,
    REPOSITORIES,
    SEQUENCE_SPACE,
    SHAP_GLOBAL,
    SHAP_VALIDATION,
    STERIC_CLASH_DISTANCE,
    STIFFNESS_NORMAL_SCALE,
)

# --- Page Setup ---

st.set_page_config(
    page_title="HYGEIA — Designing new antibiotics",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded",
)
theme.inject(st)


@st.cache_resource(show_spinner="Starting the screening model...")
def get_gatekeeper() -> hm.Gatekeeper:
    """Cached so the model is built once per server, not once per click."""
    return hm.load_gatekeeper()


GK = get_gatekeeper()

st.session_state.setdefault("pool", [])


def plot(fig: go.Figure, key: str, height: int = 430, **kwargs):
    """
    Render a chart, returning the click payload when this Streamlit supports
    point selection and None when it doesn't.
    """
    fig.update_layout(height=height)
    try:
        return st.plotly_chart(
            fig, use_container_width=True, key=key,
            on_select="rerun", selection_mode=("points",), **kwargs,
        )
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, key=key, **kwargs)
        return None


# --- Header ---

theme.utility_bar(st, [
    PROJECT["institution"],
    "DepEd Division Science &amp; Technology Fair 2026",
    PROJECT["category"],
])

theme.masthead(
    st,
    wordmark="HYGEIA",
    tagline="Nurturing the Breakthrough",
    right_html=(
        "<strong>An in-silico antibiotic discovery pipeline</strong><br>"
        "Targeting drug-resistant <em>Klebsiella pneumoniae</em>"
    ),
)

theme.hero(
    st,
    eyebrow="",
    headline="Finding antibiotics a superbug <em>cannot</em> resist",
    sub=(
        "Some infections no longer respond to any antibiotic we have. HYGEIA "
        "designs entirely new germ-killing molecules by computer, then screens "
        "them for the ones most likely to work and least likely to harm the "
        "patient — so the laboratory only has to test the very best few."
    ),
    byline=(
        "<b>" + "</b> · <b>".join(PROJECT["authors"]) + "</b><br>"
        f"{PROJECT['institution']} · {PROJECT['year']}"
    ),
)

theme.stat_band(st, [
    ("1.1×10³⁹", "Possible molecules"),
    ("9", "Designed here"),
    ("2", "Final candidates"),
    (f"{CLASSIFIER_METRICS['AUROC'][0]:.3f}", "Model accuracy"),
    ("54", "Docking runs"),
    ("3", "Bacterial targets"),
])

theme.gap(st, 38)

theme.pillars(st, [
    ("01", "The problem",
     "Bacteria are learning to survive our antibiotics. Infections that were "
     "once routine are becoming hard to treat, and very few genuinely new "
     "drugs are being found to replace the ones that stop working."),
    ("02", "Our approach",
     "We design brand-new molecules on a computer and test them in "
     "simulation. That is millions of times faster and cheaper than making "
     "every one by hand, so we can search a space far too large to explore "
     "at the bench."),
    ("03", "The result",
     "Nine new molecules, narrowed to two that are predicted to kill the "
     "bacterium without harming human cells — a shortlist a laboratory could "
     "pick up and test."),
])

theme.gap(st, 30)

theme.goddess_panel(
    st,
    eyebrow="Why we are called HYGEIA",
    title="The goddess who kept people from falling ill",
    body_html=(
        "<p>In Greek myth, <b>Asclepius</b> healed the sick. His daughter "
        "<b>Hygeia</b> did something harder: she kept people from getting "
        "sick in the first place. Her emblem — a bowl with a serpent coiled "
        "around it — still marks pharmacies all over the world, and the word "
        "<b>hygiene</b> comes from her name.</p>"
        "<p>That is the stance this project takes. Drug resistance is a "
        "problem you get ahead of or not at all; by the time an infection "
        "stops responding, the time to act has passed. "
        "<b>Nurturing the breakthrough</b> means doing the patient, "
        "unglamorous work now, so that the medicine exists on the day "
        "somebody needs it.</p>"
    ),
)

theme.gap(st, 34)

# --- Sidebar ---

with st.sidebar:
    st.markdown("## 🎀 Controls")
    st.caption("Change these and every chart updates.")

    if GK.is_surrogate:
        st.warning(
            "**Stand-in model running.** The trained model file isn't included "
            "with this site, so a stand-in was built from the same rules the "
            "real one learned. The nine peptides from the study still show "
            "their real published scores.",
            icon="⚠️",
        )
    else:
        st.success("Trained model loaded.", icon="✅")

    theme.rule(st, "Conditions")
    ph = st.slider(
        "Acidity (pH)", 5.0, 9.0, 7.0, 0.1,
        help="How acidic the surroundings are. Charge depends on it. "
             "pH 7 is neutral, like blood — that's what the study used.",
    )

    theme.rule(st, "What counts as good")
    st.caption("The four standards a peptide should meet.")
    p_min = st.slider("Minimum activity score", 0.0, 1.0,
                      ACPE_THRESHOLDS["p_active_min"], 0.01,
                      help="The screening cut-off. Peptides scoring below "
                           "this are set aside, not ruled out.")
    ai_min = st.slider("Must be at least this sturdy", 0.0, 140.0,
                       ACPE_THRESHOLDS["ai_min"], 1.0,
                       help="Aliphatic index — higher means it holds its shape "
                            "better when warm, but clumps more easily.")
    tox_max = st.slider("Must be no more toxic than", 0.0, 1.0,
                        ACPE_THRESHOLDS["toxicity_max"], 0.01,
                        help="Predicted harm to human cells. Lower is better.")
    hem_max = st.slider("Must burst no more blood cells than", 0.0, 1.0,
                        ACPE_THRESHOLDS["hemolysis_max"], 0.01,
                        help="Predicted damage to red blood cells. Lower is better.")

    thresholds = {
        "p_active_min": p_min,
        "p_active_high_conf": ACPE_THRESHOLDS["p_active_high_conf"],
        "ai_min": ai_min,
        "toxicity_max": tox_max,
        "hemolysis_max": hem_max,
    }

    theme.rule(st, "How good is the model?")
    st.metric("Overall accuracy (AUROC)", f"{CLASSIFIER_METRICS['AUROC'][0]:.4f}",
              help="Given one peptide that works and one that doesn't, the "
                   "model ranks them correctly 88% of the time.")
    st.metric("Balanced score (MCC)", f"{CLASSIFIER_METRICS['MCC'][0]:.4f}",
              help="A fair score even when most peptides in the data don't "
                   "work. 0 would be random guessing, 1 would be perfect.")
    st.caption(
        f"Tested on {sum(CONFUSION.values())} peptides it had never seen: "
        f"{CONFUSION['TP']} correctly called active, {CONFUSION['TN']} correctly "
        f"called inactive, {CONFUSION['FP'] + CONFUSION['FN']} missed."
    )

    with st.expander("🧪 How it was built"):
        for stage, tool, detail in PIPELINE_STAGES:
            st.markdown(f"**{stage}** · `{tool}`  \n<small>{detail}</small>",
                        unsafe_allow_html=True)

    with st.expander("📚 Where the data came from"):
        for name, url in REPOSITORIES.items():
            st.markdown(f"- [{name}]({url})")
        st.caption("All open, public databases.")

# --- Tabs ---

tab1, tab2, tab3, tab4 = st.tabs([
    "🧬  Design peptides",
    "💗  Why it scored that way",
    "🎯  Picking the best",
    "🔬  Shape & docking",
])

# --- Tab 1 ---
with tab1:
    theme.heading(
        st,
        "Design new peptides",
        "De novo generation and two-layer gate screening",
        "A peptide is a short chain of amino acids. Change the order and you "
        "get a different molecule. There are about "
        f"{SEQUENCE_SPACE:.0e} possible chains between 10 and 30 links long — "
        "far too many to test in a lab. So we build new ones here, then throw "
        "away the ones that can't possibly work before anyone picks up a "
        "pipette.",
    )

    left, right = st.columns([1, 1.45], gap="large")

    with left:
        theme.rule(st, "Build a batch")

        template = st.text_input(
            "Start with these letters (optional)",
            value="",
            placeholder="e.g. NGWNGNR",
            help="Each letter is one amino acid. Anything you type stays at "
                 "the front and the rest is filled in. Leave it blank to "
                 "build from scratch.",
        )

        c1, c2 = st.columns(2)
        with c1:
            len_lo = st.slider("Shortest", 10, 30, 18)
        with c2:
            len_hi = st.slider("Longest", 10, 30, 28)

        temperature = st.slider(
            "Adventurousness", 0.20, 2.00, 1.00, 0.05,
            help="Low means play it safe and stick close to what already "
                 "works. High means try unusual combinations — more variety, "
                 "but fewer good ones.",
        )
        batch = st.slider("How many to build", 3, 40, 9,
                          help="The study made nine.")
        seed = st.number_input("Repeat number", 0, 10_000, 42, 1,
                               help="Same number gives the same batch again, "
                                    "so results can be reproduced.")

        theme.rule(st, "Screening rules")
        st.caption(
            "Two quick checks. Anything failing either is dropped before the "
            "model even looks at it."
        )
        g1, g2 = st.columns(2)
        with g1:
            min_charge = st.slider(
                "Least positive charge", 0.0, 10.0, 2.0, 0.25,
                help="Bacteria have negatively charged skins. A positive "
                     "peptide is pulled towards them, like a magnet.",
            )
        with g2:
            min_mu = st.slider(
                "Least lopsidedness", 0.0, 0.45, 0.13, 0.01,
                help="The peptide needs its greasy parts bunched on one side "
                     "so it can slip into the bacterial membrane. Evenly "
                     "mixed peptides can't get in.",
            )
        st.caption(
            f"That lopsidedness setting is **{min_mu:.2f}** here, which is "
            f"**{min_mu * hf.MOMENT_SCALE_RATIO:.2f}** in the units used in "
            "the paper. Both new and published peptides are measured the same "
            "way, so the comparison is fair."
        )

        run = st.button("🌸  Build and screen", use_container_width=True)
        load_paper = st.button("📄  Show the study's nine peptides",
                               use_container_width=True)

    if run:
        seqs = hm.generate_sequences(
            n=int(batch),
            length_range=(int(len_lo), int(len_hi)),
            temperature=float(temperature),
            template=template,
            seed=int(seed),
        )
        st.session_state["pool"] = hm.apply_gate(
            seqs, GK, min_charge=float(min_charge), min_mu_h=float(min_mu)
        )

    if load_paper:
        st.session_state["pool"] = hm.apply_gate(
            [c["sequence"] for c in CANDIDATES], GK,
            min_charge=float(min_charge), min_mu_h=float(min_mu),
        )

    pool = st.session_state["pool"]

    with right:
        if not pool:
            st.info(
                "Press **Build and screen** to make a fresh batch, or **Show "
                "the study's nine peptides** to see the ones from the paper.",
                icon="🌷",
            )
        else:
            passed = [r for r in pool if r["passed"]]
            theme.rule(st, "Results")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Built", len(pool))
            m2.metric("Passed screening", len(passed),
                      delta=f"{len(passed)/len(pool):.0%} kept")
            if passed:
                best = max(passed, key=lambda r: r["p_active"])
                m3.metric("Best score", f"{best['p_active']:.3f}")
                m4.metric("Average charge",
                          f"{np.mean([r['charge'] for r in passed]):+.2f}")

            df = pd.DataFrame([{
                "Name": r["published_id"] or "new",
                "Letters": r["sequence"],
                "Numbers from": "the paper"
                                if r["descriptor_source"] == "manuscript"
                                else "calculated here",
                "Length": int(r["length"]),
                "Score": r["p_active"],
                "Charge": round(r["charge"], 2),
                "Lopsidedness": round(r["mu_h"], 3),
                "Greasiness": round(r["gravy"], 2),
                "Sturdiness": round(r["ai"], 1),
                "Clumping risk": r["aggregation"]
                                 .replace("Low aggregation risk", "Low")
                                 .replace("High aggregation risk", "High")
                                 .replace("Normal behaviour", "Normal"),
                "Passed?": "Yes" if r["passed"] else "No",
            } for r in pool]).sort_values("Score", ascending=False,
                                          na_position="last")

            st.dataframe(
                df, use_container_width=True, hide_index=True,
                column_config={
                    "Score": st.column_config.ProgressColumn(
                        "Score", min_value=0.0, max_value=1.0, format="%.3f",
                        help="The model's confidence that this peptide is active, from 0 to 1. A screening score, not a measured kill rate.",
                    ),
                },
            )

            rejected = [r for r in pool if not r["passed"]]
            if rejected:
                with st.expander(f"⚠️ {len(rejected)} didn't pass — see why"):
                    for r in rejected:
                        st.markdown(
                            f'<div class="hygeia-seq">{r["sequence"] or "(empty)"}</div>',
                            unsafe_allow_html=True,
                        )
                        for reason in r["reasons"]:
                            st.caption(f"↳ {reason}")

    # ---- The score card
    if pool:
        theme.rule(st, "Look at one peptide")

        options = [r["sequence"] for r in pool if r["passed"]]
        if not options:
            st.warning("Nothing passed screening, so there's nothing to score. "
                       "Try loosening the screening rules on the left.")
        else:
            labels = {
                r["sequence"]: f"{r['published_id'] or 'New peptide'} — {r['sequence'][:22]}…"
                for r in pool if r["passed"]
            }
            chosen = st.selectbox("Pick one", options,
                                  format_func=lambda s: labels[s],
                                  key="tab1_candidate")
            rec = next(r for r in pool if r["sequence"] == chosen)
            d, d_source = hm.descriptors_for(chosen, ph=ph)

            cc1, cc2 = st.columns([1, 1.9], gap="large")
            with cc1:
                tier = ("Strong signal" if rec["p_active"] >= 0.80
                        else "Above the screening cut-off" if rec["p_active"] >= 0.50
                        else "Below the screening cut-off")
                tier_kind = "pass" if rec["p_active"] >= 0.50 else "warn"
                clump = rec["aggregation"]
                st.markdown(
                    f"""
                    <div class="hygeia-verdict">
                        <div class="cap">Activity score</div>
                        <div class="big">{rec['p_active']:.3f}</div>
                        <div class="cap">model confidence, 0 to 1</div>
                        <div style="margin-top:11px">
                            {theme.chip(tier, tier_kind)}
                            {theme.chip(f"Clumping: {clump.split()[0]}",
                                        'warn' if 'High' in clump else 'pass')}
                            {theme.chip(rec['published_id'], 'lead')
                             if rec['published_id'] else ''}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown(f'<div class="hygeia-seq">{chosen}</div>',
                            unsafe_allow_html=True)

            with cc2:
                g1, g2, g3 = st.columns(3)
                g1.metric("Positive charge", f"{d['charge']:+.2f}",
                          help="Pulls the peptide towards the bacteria. "
                               "Higher is generally better.")
                g2.metric("Lopsidedness", f"{d['mu_h']:.3f}",
                          help="How well the greasy parts bunch on one side.")
                g3.metric("Greasiness", f"{d['gravy']:+.2f}",
                          help="Overall oiliness. Too high and it sticks to "
                               "human cells too.")
                g4, g5, g6 = st.columns(3)
                g4.metric("Sturdiness", f"{d['ai']:.1f}",
                          help="Holds its shape when warm — but a high number "
                               "also means it clumps together more easily.")
                g5.metric("Charge balance point", f"{d['pI']:.2f}",
                          help="The acidity at which the peptide has no "
                               "overall charge.")
                g6.metric("Length", f"{int(d['length'])} links")
                st.caption(
                    "These are the published numbers."
                    if d_source == "manuscript" else
                    "Calculated live from the formulas in the paper."
                )

                counts = pd.Series(list(chosen)).value_counts().reindex(
                    list(hf.CANONICAL_AA), fill_value=0
                )
                fig = go.Figure(go.Bar(
                    x=counts.index.tolist(), y=counts.values.tolist(),
                    marker=dict(color=counts.values.tolist(),
                                colorscale=theme.ROSE_SEQUENTIAL,
                                line=dict(color=theme.SURFACE, width=2)),
                    text=[c if c else "" for c in counts.values],
                    textposition="outside",
                    hovertemplate="%{x}: %{y}<extra></extra>",
                ))
                fig.update_layout(
                    title="What it's made of",
                    yaxis_title="How many", xaxis_title="Amino acid",
                    showlegend=False, bargap=0.28,
                )
                plot(fig, key="composition", height=260)

        with st.expander("📐 Show the maths behind these numbers"):
            st.markdown(
                r"""
**Positive charge** — every peptide carries acidic and basic groups that gain
or lose charge depending on the acidity around them. The fraction carrying
charge at a given pH follows Henderson-Hasselbalch:

$$z = \sum_{\text{basic}} \frac{1}{1 + 10^{\,pH - pK_a}} - \sum_{\text{acidic}} \frac{1}{1 + 10^{\,pK_a - pH}}$$

That's why the paper reports charges like +4.975 rather than whole numbers.
Charge is the single most important feature the model uses.

**Lopsidedness (hydrophobic moment)** — each amino acid is placed around a
spiral at $100^\circ$ per step and weighted by how greasy it is. The arrows
only add up to something large when the greasy ones land on the same side:

$$\mu_H = \frac{1}{N}\sqrt{\Big(\sum_n H_n \sin \delta_n\Big)^2 + \Big(\sum_n H_n \cos \delta_n\Big)^2}$$

**Greasiness (GRAVY)** — $\frac{1}{L}\sum_i h_i$, the plain average over all
links, with no regard to where they sit.

**Sturdiness (aliphatic index)** —
$\text{AI} = X_{Ala} + 2.9\,X_{Val} + 3.9\,(X_{Ile} + X_{Leu})$, weighted by
how bulky each side branch is.

**Charge balance point (pI)** — the acidity at which $z = 0$. There's no way to
solve for it directly, so it's found by repeated halving of the range.
                """
            )

# --- Tab 2 ---
with tab2:
    theme.heading(
        st,
        "Why it scored that way",
        "Explainable AI — TreeSHAP attribution",
        "A model that gives an answer without a reason is hard to trust. This "
        "page opens it up: which features the model leans on, which way each "
        "one pushes, and how much each one moved the score for a single "
        "peptide.",
    )

    theme.rule(st, "What the model pays attention to")
    gl, gr = st.columns([1.5, 1], gap="large")

    with gl:
        labels = [s[0] for s in SHAP_GLOBAL][::-1]
        values = [s[1] for s in SHAP_GLOBAL][::-1]
        kinds = [s[2] for s in SHAP_GLOBAL][::-1]

        fig = go.Figure()
        for kind, colour, name in [
            ("physicochemical", theme.CATEGORICAL[0], "Physical property"),
            ("embedding", theme.CATEGORICAL[1], "Pattern learned from data"),
        ]:
            mask = [k == kind for k in kinds]
            fig.add_trace(go.Bar(
                y=[l for l, m in zip(labels, mask) if m],
                x=[v for v, m in zip(values, mask) if m],
                name=name, orientation="h",
                marker=dict(color=colour, line=dict(color=theme.SURFACE, width=2)),
                text=[f"{v:.4f}" for v, m in zip(values, mask) if m],
                textposition="outside",
                hovertemplate="%{y}<br>importance %{x:.4f}<extra></extra>",
            ))
        fig.update_layout(
            title="How much each feature sways the answer",
            xaxis_title="Average influence on the score",
            barmode="overlay", bargap=0.35,
            xaxis=dict(range=[0, max(values) * 1.3]),
        )
        plot(fig, key="shap_global", height=520)

    with gr:
        theme.card(st, "The headline result", """
        <p><b>Positive charge matters most</b> — 61% more than anything else.
        Lopsidedness comes fourth.</p>
        <p>Here's why that's exciting. Nobody told the model any biology. It
        was handed numbers and outcomes and left to find its own patterns. On
        its own it landed on the <b>same two properties</b> scientists have
        known for decades are what let these molecules break into bacteria.</p>
        <p>That's a sign it learned something real about how the molecules
        work, instead of memorising quirks of the dataset.</p>
        """)
        st.metric("Biggest single penalty", f"{MAX_NEGATIVE_PHI:.2f}",
                  help="Peptides whose greasy parts are spread evenly get hit "
                       "harder than anything else in the model.")

    theme.rule(st, "Checking those patterns are real")
    vdf = pd.DataFrame(
        [(f, r, rho, note) for f, r, rho, note in SHAP_VALIDATION],
        columns=["Feature", "Straight-line fit", "Ranking fit", "What it means"],
    )
    st.dataframe(vdf, use_container_width=True, hide_index=True)
    st.caption(
        "**Charge behaves simply** — the two fits agree, so more charge means "
        "proportionally more activity, all the way up. "
        "**Lopsidedness doesn't** — the ranking fit is much stronger than the "
        "straight-line fit, which means it works like a pass mark rather than "
        "a dial. Below a certain point the peptide simply can't get into the "
        "membrane and the model punishes it hard; above that point, extra "
        "lopsidedness adds little. Both patterns hold up statistically."
    )

    theme.rule(st, "One peptide, broken down")
    st.markdown(
        '<p class="hygeia-lede">Every feature either pushes the score up or '
        'drags it down, starting from an average peptide. The bars add up '
        'exactly to the final answer.</p>',
        unsafe_allow_html=True,
    )

    pool = st.session_state["pool"]
    sources = ([(r["published_id"] or r["sequence"][:18], r["sequence"])
                for r in pool if r["passed"]]
               or [(c["id"], c["sequence"]) for c in CANDIDATES])

    pick = st.selectbox("Peptide", sources, format_func=lambda t: t[0],
                        key="tab2_pick")
    seq = pick[1]

    with st.spinner("Working out each feature's share..."):
        phi, base, method = hm.local_attribution(GK, seq)

    ll, lr = st.columns([1.6, 1], gap="large")

    with ll:
        items = sorted(phi.items(), key=lambda kv: abs(kv[1]), reverse=True)[:10]
        items = items[::-1]
        names = [hf.FEATURE_LABELS.get(k, k) for k, _ in items]
        vals = [v for _, v in items]

        fig = go.Figure(go.Waterfall(
            orientation="h", y=names, x=vals,
            measure=["relative"] * len(vals), base=base,
            increasing=dict(marker=dict(color=theme.CATEGORICAL[0])),
            decreasing=dict(marker=dict(color=theme.CATEGORICAL[1])),
            connector=dict(line=dict(color=theme.PETAL, width=1.5)),
            text=[f"{v:+.4f}" for v in vals], textposition="outside",
            hovertemplate="%{y}<br>moves the score %{x:+.4f}<extra></extra>",
        ))
        fig.add_vline(x=base, line=dict(color=theme.INK_MUTED, dash="dot", width=1.5),
                      annotation_text=f"average peptide: {base:.3f}",
                      annotation_position="top")
        fig.update_layout(
            title=f"What moved the score — {pick[0]}",
            xaxis_title="Pushes the score up (pink) or down (teal)",
        )
        plot(fig, key="shap_local", height=470)

    with lr:
        total = base + sum(phi.values())
        st.metric("Starting point", f"{base:.4f}",
                  help="The score an average peptide would get before looking "
                       "at this one's features.")
        st.metric("All the pushes added up", f"{sum(phi.values()):+.4f}")
        st.metric("Final score", f"{total:.4f}")
        st.caption(
            f"Worked out using **{method}**. The pushes always add up exactly "
            "to the final score — that's a mathematical guarantee of this "
            "method, not a coincidence."
        )
        st.markdown(f'<div class="hygeia-seq">{seq}</div>', unsafe_allow_html=True)

        with st.expander("📐 Show the maths"):
            st.markdown(
                r"""
Each feature's share is its **Shapley value**, borrowed from game theory —
originally a way to split winnings fairly among players who contributed
different amounts to a team:

$$\phi_j = \sum_{S \subseteq F \setminus \{j\}} \frac{|S|!\,(|F|-|S|-1)!}{|F|!}\big[f_x(S \cup \{j\}) - f_x(S)\big]$$

- $F$ — all the features
- $S$ — any group of features that leaves out feature $j$
- $f_x(S)$ — the score using only the features in $S$

The bracket is how much the score changes when feature $j$ joins group $S$. The
fraction in front weights every possible group so no feature gets credit just
for being considered early. A positive $\phi_j$ pushes towards "this works".

TreeSHAP computes this exactly for forest models without having to try all
orderings, which would be impossible.
                """
            )

    theme.rule(st, "Was combining two kinds of feature worth it?")
    adf = pd.DataFrame(FEATURE_ABLATION, columns=[
        "What the model was given", "Number of features", "Practice accuracy",
        "Practice variation", "Test accuracy", "Balanced score",
        "Overall accuracy", "Precision score",
    ])
    st.dataframe(adf, use_container_width=True, hide_index=True)
    st.caption(
        "Yes. Physical properties alone and learned patterns alone both score "
        "about 0.56 on the balanced measure; together they reach 0.61. "
        "Trimming from 172 features down to the 30 that matter made it better "
        "still, and steadier between runs — proof the discarded features were "
        "noise, not signal."
    )

# --- Tab 3 ---
with tab3:
    theme.heading(
        st,
        "Picking the best few",
        "Multi-objective Pareto optimisation (ACPE)",
        "Four things matter at once, and they fight each other. We want a "
        "peptide that kills bacteria well, holds its shape, doesn't poison "
        "human cells and doesn't burst red blood cells. The changes that make "
        "one better usually make another worse — so there is no single "
        "winner, only a set of different compromises.",
    )

    ranked = hm.acpe_rank(CANDIDATES, thresholds)
    pdf = pd.DataFrame(ranked)

    s1, s2 = st.columns(2)
    s1.metric("Peptides considered", len(ranked))
    s2.metric("Worth considering", int(pdf["non_dominated"].sum()),
              help="No other peptide beats these on all four counts at once.")

    theme.rule(st, "The trade-off map")
    st.markdown(
        '<p class="hygeia-lede">Each dot is one peptide. Pick what goes on '
        'each axis and see how they compare. Click a dot to inspect it.</p>',
        unsafe_allow_html=True,
    )

    axis_options = {
        "Activity score": "p_active",
        "Sturdiness": "ai",
        "Toxic to human cells": "toxicity",
        "Bursts red blood cells": "hemolysis",
        "Positive charge": "charge",
        "Lopsidedness": "mu_h",
        "Greasiness": "gravy",
    }

    cc = st.columns([1, 1, 1, 1.1])
    with cc[0]:
        x_label = st.selectbox("Across the bottom", list(axis_options), index=0)
    with cc[1]:
        y_label = st.selectbox("Up the side", list(axis_options), index=3)
    with cc[2]:
        z_label = st.selectbox("Depth (3-D only)", list(axis_options), index=1)
    with cc[3]:
        view = st.radio("View", ["Flat", "3-D"], horizontal=True)

    xk, yk, zk = axis_options[x_label], axis_options[y_label], axis_options[z_label]

    nd = [r for r in ranked if r["non_dominated"]]
    dom = [r for r in ranked if not r["non_dominated"]]
    leads = [r for r in ranked if r["selected_lead"]]

    hover = (
        "<b>%{customdata[0]}</b><br>%{customdata[1]}<br>"
        "Activity score %{customdata[2]:.3f}<br>Sturdiness %{customdata[3]:.1f}<br>"
        "Toxicity %{customdata[4]:.3f}<br>Blood cell damage %{customdata[5]:.3f}"
        "<extra></extra>"
    )
    cd = lambda rs: [[r["id"], r["sequence"], r["p_active"], r["ai"],  # noqa: E731
                      r["toxicity"], r["hemolysis"]] for r in rs]

    if view == "Flat":
        fig = go.Figure()
        if dom:
            fig.add_trace(go.Scatter(
                x=[r[xk] for r in dom], y=[r[yk] for r in dom],
                mode="markers+text", name="Beaten by another peptide",
                marker=dict(size=14, color=theme.CATEGORICAL[1], symbol="x-thin",
                            line=dict(color=theme.CATEGORICAL[1], width=2.5)),
                text=[r["id"].replace("Peptide_", "P") for r in dom],
                textposition="bottom center",
                textfont=dict(size=10, color=theme.INK_MUTED),
                customdata=cd(dom), hovertemplate=hover,
            ))
        fig.add_trace(go.Scatter(
            x=[r[xk] for r in nd], y=[r[yk] for r in nd],
            mode="markers+text", name="Worth considering",
            marker=dict(size=16, color=theme.CATEGORICAL[0], symbol="circle",
                        line=dict(color=theme.SURFACE, width=2)),
            text=[r["id"].replace("Peptide_", "P") for r in nd],
            textposition="top center",
            textfont=dict(size=11, color=theme.ROSE_DEEP),
            customdata=cd(nd), hovertemplate=hover,
        ))
        fig.add_trace(go.Scatter(
            x=[r[xk] for r in leads], y=[r[yk] for r in leads],
            mode="markers+text", name="Chosen",
            marker=dict(size=26, color="rgba(0,0,0,0)", symbol="star-diamond-open",
                        line=dict(color=theme.ROSE_DEEP, width=3)),
            text=[r["id"] for r in leads], textposition="middle right",
            textfont=dict(size=12, color=theme.ROSE_DEEP),
            customdata=cd(leads), hovertemplate=hover,
        ))

        guides = {"p_active": p_min, "ai": ai_min,
                  "toxicity": tox_max, "hemolysis": hem_max}
        if xk in guides:
            fig.add_vline(x=guides[xk], line=dict(color=theme.ROSE_LIGHT,
                          dash="dash", width=2), annotation_text="our standard")
        if yk in guides:
            fig.add_hline(y=guides[yk], line=dict(color=theme.ROSE_LIGHT,
                          dash="dash", width=2), annotation_text="our standard")

        fig.update_layout(title=f"{y_label} against {x_label}",
                          xaxis_title=x_label, yaxis_title=y_label)
        event = plot(fig, key="pareto2d", height=540)
    else:
        fig = go.Figure()
        for group, name, colour, symbol in [
            (nd, "Worth considering", theme.CATEGORICAL[0], "circle"),
            (dom, "Beaten by another peptide", theme.CATEGORICAL[1], "x"),
        ]:
            if not group:
                continue
            fig.add_trace(go.Scatter3d(
                x=[r[xk] for r in group], y=[r[yk] for r in group],
                z=[r[zk] for r in group], mode="markers+text", name=name,
                marker=dict(size=9, color=colour, symbol=symbol,
                            line=dict(color=theme.SURFACE, width=1.5)),
                text=[r["id"].replace("Peptide_", "P") for r in group],
                textfont=dict(size=10, color=theme.INK_SOFT),
                customdata=cd(group), hovertemplate=hover,
            ))
        if leads:
            fig.add_trace(go.Scatter3d(
                x=[r[xk] for r in leads], y=[r[yk] for r in leads],
                z=[r[zk] for r in leads], mode="markers+text", name="Chosen",
                marker=dict(size=15, color=theme.ROSE_DEEP, symbol="diamond",
                            line=dict(color=theme.SURFACE, width=2)),
                text=[r["id"] for r in leads],
                textfont=dict(size=11, color=theme.ROSE_DEEP),
                customdata=cd(leads), hovertemplate=hover,
            ))
        fig.update_layout(title=f"{x_label} / {y_label} / {z_label}",
                          scene=dict(xaxis_title=x_label, yaxis_title=y_label,
                                     zaxis_title=z_label))
        theme.style_3d(fig)
        event = plot(fig, key="pareto3d", height=600)

    clicked = None
    try:
        pts = (event or {}).get("selection", {}).get("points", [])
        if pts and pts[0].get("customdata"):
            clicked = pts[0]["customdata"][0]
    except Exception:
        clicked = None

    theme.rule(st, "Inspect one")
    ids = [r["id"] for r in ranked]
    default_idx = ids.index(clicked) if clicked in ids else ids.index(FINAL_LEADS[0])
    focus_id = st.selectbox("Peptide (or click a dot above)", ids,
                            index=default_idx, key="tab3_focus")
    rec = next(r for r in ranked if r["id"] == focus_id)

    il, ir = st.columns([1, 1.5], gap="large")
    with il:
        plain = {
            "Activity floor": "Above activity cut-off",
            "Stability floor": "Sturdy enough",
            "Toxicity ceiling": "Safe enough for cells",
            "Hemolysis ceiling": "Safe enough for blood",
        }
        chips = "".join(
            theme.chip(f"{plain.get(k, k)}: {'yes' if v else 'no'}",
                       "pass" if v else "fail")
            for k, v in rec["threshold_checks"].items()
        )
        verdict = ("Selected lead" if rec["selected_lead"]
                   else "Worth considering" if rec["non_dominated"]
                   else "Beaten by another peptide")
        kind = ("pass" if rec["selected_lead"]
                else "warn" if rec["non_dominated"] else "fail")
        risk_chip = (
            theme.chip(f"Safest #{rec['risk_rank']} of {len(nd)}",
                       "pass" if rec["risk_rank"] <= 2 else "")
            if rec["risk_rank"] else ""
        )
        st.markdown(
            f"""
            <div class="hygeia-verdict">
                <div class="cap">{rec['id']}</div>
                <div class="big">{rec['p_active']:.3f}</div>
                <div class="cap">activity score, 0 to 1</div>
                <div style="margin-top:11px">
                    {theme.chip(verdict, kind)}{risk_chip}
                </div>
                <div style="margin-top:7px">{chips}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(f'<div class="hygeia-seq">{rec["sequence"]}</div>',
                    unsafe_allow_html=True)

    with ir:
        ai_max = max(r["ai"] for r in ranked)
        bars = [
            ("Activity score", rec["p_active"], "want high"),
            ("Sturdiness", rec["ai"] / ai_max, "want high"),
            ("Toxic to cells", rec["toxicity"], "want low"),
            ("Bursts blood cells", rec["hemolysis"], "want low"),
        ]
        fig = go.Figure()
        for direction, colour, nm in [
            ("want high", theme.CATEGORICAL[0], "Higher is better"),
            ("want low", theme.CATEGORICAL[1], "Lower is better"),
        ]:
            sel = [(n, v) for n, v, dd in bars if dd == direction]
            fig.add_trace(go.Bar(
                y=[n for n, _ in sel], x=[v for _, v in sel], name=nm,
                orientation="h",
                marker=dict(color=colour, line=dict(color=theme.SURFACE, width=2)),
                text=[f"{v:.3f}" for _, v in sel], textposition="outside",
                hovertemplate="%{y}: %{x:.3f}<extra></extra>",
            ))
        fig.update_layout(
            title=f"All four at once — {rec['id']}",
            xaxis=dict(range=[0, 1.18], title="Everything on a 0–1 scale"),
            barmode="overlay", bargap=0.4,
        )
        plot(fig, key="objective_profile", height=330)
        st.caption(
            "Sturdiness is divided by the highest in the group so it fits the "
            "same 0–1 scale as the others. Putting a 0–1 value and a 0–132 "
            "value on one chart any other way would flatten the small ones to "
            "nothing."
        )

    theme.rule(st, "Why the finalists score low on activity")
    by_id = {r["id"]: r for r in ranked}
    lead_scores = ", ".join(
        f"{i} scores {by_id[i]['p_active']:.2f}" for i in FINAL_LEADS if i in by_id
    )
    wl, wr = st.columns([1, 1], gap="large")
    with wl:
        theme.card(st, "The score is cautious by design", f"""
        <p>{lead_scores} — both under our 0.50 cut-off. That looks
        damning until you see how the model was built.</p>
        <p>It was deliberately tuned to be <b>careful rather than
        optimistic</b>: it correctly spots
        {CLASSIFIER_METRICS['Specificity'][0]:.0%} of peptides that don't work,
        but only {CLASSIFIER_METRICS['Sensitivity'][0]:.0%} of the ones that
        do. In other words it misses real working peptides far more often than
        it invents fake ones.</p>
        <p>That is the right way round for screening — a false alarm wastes
        weeks of lab time, a near miss only shortens the shortlist. But it
        means a low score is <b>weak evidence against</b> a peptide, not proof
        it fails.</p>
        """)
    with wr:
        theme.card(st, "Two other lines of evidence say otherwise", """
        <p><b>Docking.</b> Both finalists stuck to the bacterial proteins about
        as well as peptides already <i>proven</i> to kill this bacterium — and
        they were designed without ever being shown what the target looks like.
        That test doesn't use the activity model at all, so it's independent
        evidence.</p>
        <p><b>Movement.</b> Peptide_03 held its grip on all three targets when
        shaken, and its predicted flexing matched real X-ray measurements.</p>
        <p>So the honest summary is: the screening model is unsure about these
        two, while the structural work supports them. That disagreement is
        exactly what laboratory testing is for.</p>
        """)

    st.info(
        "**Seven of the nine are worth considering** — no single one of them "
        "is beaten outright. So being on the list isn't enough to choose by.\n\n"
        "And here's the catch: **not one peptide meets all four standards at "
        "once.** The changes that make a peptide kill bacteria better also "
        "make it more dangerous to human cells. That conflict is the real "
        "finding — it's exactly why you can't just add the four scores "
        "together and pick the top one.\n\n"
        "So the choice was made on **safety first**: among the seven, take the "
        "two least likely to harm human cells. That gives **Peptide_07** and "
        "**Peptide_03**. A gentler peptide that works moderately well is a "
        "better starting point than a strong one that also kills the patient's "
        "cells — you can improve a safe molecule, but you can't easily make a "
        "toxic one safe.\n\n"
        "Picking on killing power alone would have chosen Peptide_01, the one "
        "most likely to clump uselessly in the bottle.",
        icon="🎀",
    )

    theme.rule(st, "Every peptide, side by side")
    show = pdf[["id", "sequence", "p_active", "ai", "toxicity", "hemolysis",
                "risk_score", "non_dominated", "n_passed", "selected_lead"]].copy()
    show.columns = ["Name", "Letters", "Activity score", "Sturdiness",
                    "Toxic to cells", "Bursts blood cells", "Total risk",
                    "Worth considering", "Standards met", "Chosen"]
    st.dataframe(
        show.sort_values(["Chosen", "Worth considering", "Total risk"],
                         ascending=[False, False, True]),
        use_container_width=True, hide_index=True,
        column_config={
            "Standards met": st.column_config.ProgressColumn(
                "Standards met", min_value=0, max_value=4, format="%d of 4",
            ),
        },
    )
    st.caption(
        "A head-to-head statistical test between the chosen and the rest was "
        "left out on purpose: with only nine peptides, even a perfect result "
        "couldn't clear the significance bar. Reporting one would have been "
        "misleading."
    )

# --- Tab 4 ---
with tab4:
    theme.heading(
        st,
        "Shape and docking",
        "Molecular docking and elastic network modelling",
        "Knowing a peptide should work isn't the same as knowing it fits. "
        "Here the designed peptides are fitted against three proteins on the "
        "surface of the bacterium, and then shaken to see whether the fit "
        "survives being jostled by ordinary heat.",
    )

    # ---- The 3-D viewer
    theme.rule(st, "See the docked complex in 3-D")

    bundled = hs.list_bundled()
    sources = []
    if bundled:
        sources.append("From this site")
    sources += ["Upload my own file", "Just the bacterial protein (from the PDB)"]

    vc = st.columns([1.1, 1, 1, 0.9])
    with vc[0]:
        source = st.selectbox("Structure to show", sources, key="struct_source")
    with vc[1]:
        style = st.selectbox("Style", list(hs.STYLE_CHOICES),
                             index=1, key="struct_style")
    with vc[2]:
        if source == "Just the bacterial protein (from the PDB)":
            target_name = st.selectbox("Protein", list(hs.RECEPTOR_IDS),
                                       key="struct_target")
        else:
            target_name = None
    with vc[3]:
        spin = st.checkbox("Spin it", value=False, key="struct_spin")

    structure = None
    load_error = None

    if source == "From this site":
        chosen_file = st.selectbox("File", bundled, key="struct_file")
        try:
            structure = hs.load_bundled(chosen_file)
        except Exception as exc:
            load_error = str(exc)

    elif source == "Upload my own file":
        up = st.file_uploader(
            "Drop a docked complex here (.pdb or .cif)",
            type=["pdb", "ent", "cif"], key="struct_upload",
        )
        if up is not None:
            text = up.getvalue().decode("utf-8", errors="replace")
            structure = hs.Structure(label=up.name, text=text,
                                     fmt="mmcif" if up.name.lower().endswith(
                                         (".cif", ".mmcif")) else "pdb",
                                     source="uploaded")

    if load_error:
        st.error(load_error)

    if source == "Just the bacterial protein (from the PDB)":
        pdb_id = hs.RECEPTOR_IDS[target_name]
        components.html(
            hs.rcsb_viewer_html(pdb_id, style=style, spin=spin, height=520),
            height=544, scrolling=False,
        )
        st.caption(
            f"**{target_name}** (PDB entry {pdb_id}), downloaded live from the "
            "Protein Data Bank by your browser. This is the bacterial protein "
            "on its own — no peptide attached. To see a peptide actually "
            "docked into it, load one of the HPEPDOCK result files."
        )

    elif structure is not None:
        summary = hs.summarise(structure)
        peptide_chain = None
        if summary["chains"]:
            chain_list = list(summary["chains"])
            guess = summary["likely_peptide"]
            peptide_chain = st.selectbox(
                "Which chain is the peptide?", chain_list,
                index=chain_list.index(guess) if guess in chain_list else 0,
                key="struct_chain",
                help="The peptide is drawn in pink, the bacterial protein in "
                     "blue. The shortest chain is usually the peptide.",
            )
        try:
            components.html(
                hs.viewer_html(structure, style=style,
                               peptide_chain=peptide_chain, spin=spin, height=520),
                height=544, scrolling=False,
            )
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("File", structure.label)
            mc2.metric("Atoms", f"{summary['atoms']:,}")
            mc3.metric("Chains", ", ".join(summary["chains"]) or "—")
            st.caption(
                "Pink is your designed peptide, blue is the bacterial protein. "
                "Drag to turn it, scroll to zoom."
            )
        except Exception as exc:
            st.error(f"Couldn't display that structure: {exc}")

    elif source == "Upload my own file":
        st.info("Choose a `.pdb` file above to see it in 3-D.", icon="🧊")

    if not bundled:
        with st.expander("💡 How to put your docked complexes on this site permanently"):
            st.markdown(
                f"""
So they're always here and nothing has to be uploaded at the fair:

1. In your GitHub repo, click **Add file → Create new file**
2. Type `{hs.STRUCTURE_DIR}/README.md` as the name — typing the `/` creates the
   folder
3. Put anything in the box (even just the word `structures`) and commit
4. Open the new `{hs.STRUCTURE_DIR}` folder, click **Add file → Upload files**,
   and drop in your HPEPDOCK result files

Name them clearly so they read well in the dropdown, for example
`Peptide03_OmpW.pdb` or `Peptide07_LptDE.pdb`. They'll appear here
automatically about a minute after you commit.
                """
            )

    # ---- Docking results
    theme.rule(st, "How well each peptide sticks")
    st.markdown(
        '<p class="hygeia-lede">Docking software tries thousands of ways a '
        'peptide could sit on a protein and scores the best fit. More negative '
        'means a tighter, more comfortable fit. Every one of the 54 '
        'combinations tested came out strongly negative.</p>',
        unsafe_allow_html=True,
    )

    tcols = st.columns(3)
    for col, (name, t) in zip(tcols, DOCKING_TARGETS.items()):
        with col:
            favours_kind = "pass" if t["favours"] == "De novo" else "warn"
            favours_text = ("Our peptides fit better" if t["favours"] == "De novo"
                            else "Known peptides fit better")
            theme.card(st, f"{name}", f"""
            <p>{t['description']}</p>
            <p><b>Our designs</b> {t['median_de_novo']:.1f}<br>
               <b>Known peptides</b> {t['median_dataset']:.1f}</p>
            {theme.chip(favours_text, favours_kind)}
            {theme.chip(f"difference: {t['magnitude']}", "")}
            <p><a href="https://www.rcsb.org/structure/{t['pdb']}" target="_blank">
               Look up {t['pdb']} in the PDB ↗</a></p>
            """)

    dl, dr = st.columns([1.5, 1], gap="large")
    with dl:
        names = list(DOCKING_TARGETS)
        fig = go.Figure()
        for key, colour, nm in [
            ("median_de_novo", theme.CATEGORICAL[0], "Our designed peptides"),
            ("median_dataset", theme.CATEGORICAL[1], "Known working peptides"),
        ]:
            iqr_key = "iqr_de_novo" if "de_novo" in key else "iqr_dataset"
            vals = [DOCKING_TARGETS[n][key] for n in names]
            lo = [abs(DOCKING_TARGETS[n][iqr_key][0] - DOCKING_TARGETS[n][key])
                  for n in names]
            hi = [abs(DOCKING_TARGETS[n][iqr_key][1] - DOCKING_TARGETS[n][key])
                  for n in names]
            fig.add_trace(go.Bar(
                x=names, y=vals, name=nm,
                marker=dict(color=colour, line=dict(color=theme.SURFACE, width=2)),
                error_y=dict(type="data", symmetric=False, array=hi, arrayminus=lo,
                             color=theme.INK_MUTED, thickness=1.5, width=6),
                text=[f"{v:.0f}" for v in vals], textposition="outside",
                hovertemplate="%{x} — %{y:.2f}<extra></extra>",
            ))
        fig.update_layout(
            title="Fit score against each bacterial protein",
            yaxis_title="More negative = tighter fit",
            xaxis_title="Bacterial surface protein", bargap=0.3, barmode="group",
        )
        plot(fig, key="docking", height=430)

    with dr:
        theme.card(st, "What this does and doesn't prove", """
        <p>Our brand-new peptides fit about as well as peptides already known
        to kill this bacterium — despite never being shown the target's shape
        while being designed. For a design pipeline, matching the real thing
        is a good result.</p>
        <p>None of the differences are statistically significant. That sounds
        disappointing, but it's a limit of <b>sample size</b>, not a finding of
        sameness. With nine peptides per group, the test can only detect
        enormous differences.</p>
        """)
        powdf = pd.DataFrame(
            [(f"{d:.1f}× typical spread", f"{p:.0%}") for d, p in DOCKING_POWER],
            columns=["If the real difference were…", "Chance of spotting it"],
        )
        st.dataframe(powdf, use_container_width=True, hide_index=True)
        st.caption(
            "The biggest difference actually seen was around 0.7× the typical "
            "spread — where the chance of detecting it was under 20%. So "
            "'no significant difference' here means 'not enough peptides to "
            "tell', not 'definitely the same'."
        )

    # ---- Flexibility
    theme.rule(st, "Does the fit survive being shaken?")
    st.markdown(
        '<p class="hygeia-lede">Docking shows one frozen snapshot. But real '
        'molecules jiggle constantly from heat. This step works out the ways '
        'each pair can wobble and how much energy each wobble costs — a cheap '
        'way to spot pairings that would fall apart before spending weeks on a '
        'full simulation.</p>',
        unsafe_allow_html=True,
    )

    ndf = pd.DataFrame(NMA_COMPLEXES)
    ndf["complex"] = ndf["peptide"].str.replace("Peptide_0", "P") + " × " + ndf["target"]

    e1, e2 = st.columns([1.35, 1], gap="large")

    with e1:
        fig = go.Figure()
        for pep, colour in [("Peptide_03", theme.CATEGORICAL[0]),
                            ("Peptide_07", theme.CATEGORICAL[2])]:
            sub = ndf[ndf["peptide"] == pep]
            fig.add_trace(go.Bar(
                x=sub["target"], y=sub["lambda_1"], name=pep,
                marker=dict(color=colour, line=dict(color=theme.SURFACE, width=2)),
                text=[f"{v:.1e}" for v in sub["lambda_1"]], textposition="outside",
                hovertemplate="%{x}<br>stiffness %{y:.3e}<extra></extra>",
            ))
        fig.update_layout(
            title="How stiff each pairing is",
            yaxis=dict(title="Stiffness (squeezed scale)", type="log"),
            xaxis_title="Bacterial protein", barmode="group", bargap=0.3,
        )
        plot(fig, key="eigen", height=400)
        st.caption(
            "A low number means the pair flexes easily; a high number means "
            "it's rigid. The interesting result: stiffness depends on **which "
            "protein**, not which peptide. The small, tightly-wound OmpW is "
            "about ten times stiffer than the larger OmpA and LptDE, no matter "
            "which peptide is attached. None of the six showed the warning "
            "signs of a structure about to fall apart. The scale is squeezed "
            "because the values span a hundredfold range."
        )

    with e2:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=ndf["complex"], x=ndf["mode1_variance"] * 100,
            name="Main wobble", orientation="h",
            marker=dict(color=theme.CATEGORICAL[0],
                        line=dict(color=theme.SURFACE, width=2)),
            text=[f"{v*100:.0f}%" for v in ndf["mode1_variance"]],
            textposition="inside", insidetextanchor="middle",
            hovertemplate="%{y}<br>main wobble: %{x:.0f}%<extra></extra>",
        ))
        fig.add_trace(go.Bar(
            y=ndf["complex"],
            x=(ndf["cum3_variance"] - ndf["mode1_variance"]) * 100,
            name="Next two wobbles", orientation="h",
            marker=dict(color=theme.CATEGORICAL[4],
                        line=dict(color=theme.SURFACE, width=2)),
            text=[f"{v*100:.0f}%" for v in ndf["cum3_variance"]],
            textposition="outside",
            hovertemplate="%{y}<br>top three combined<extra></extra>",
        ))
        fig.update_layout(
            title="How much of the movement the main wobbles explain",
            xaxis=dict(title="Share of all movement (%)", range=[0, 78]),
            barmode="stack", bargap=0.35,
        )
        plot(fig, key="variance", height=400)
        st.caption(
            "In every pairing the single biggest wobble accounts for a third "
            "of all movement, and the top three for more than half. The "
            "peptide moves **together with** the protein it's attached to "
            "rather than flapping around independently — a sign it's genuinely "
            "held in place."
        )

    theme.rule(st, "A warning sign in two of the six")
    s1c, s2c = st.columns([1.35, 1], gap="large")
    with s1c:
        fig = go.Figure()
        for flagged, colour, nm in [
            (False, theme.STATUS["good"], "Normal, sensible packing"),
            (True, theme.STATUS["critical"], "Atoms overlapping — problem"),
        ]:
            sub = ndf[ndf["steric_flag"] == flagged]
            if sub.empty:
                continue
            fig.add_trace(go.Bar(
                x=sub["complex"], y=sub["stiffness_max"], name=nm,
                marker=dict(color=colour, line=dict(color=theme.SURFACE, width=2)),
                text=[f"{v:,.0f}" for v in sub["stiffness_max"]],
                textposition="outside",
                hovertemplate="%{x}<br>%{y:,.0f}<extra></extra>",
            ))
        fig.add_hline(y=STIFFNESS_NORMAL_SCALE,
                      line=dict(color=theme.ROSE, dash="dash", width=2),
                      annotation_text=f"normal is {STIFFNESS_NORMAL_SCALE:.0f}")
        fig.update_layout(
            title="Strongest spring in the model",
            yaxis=dict(title="Spring strength (squeezed scale)", type="log"),
            xaxis_title="Pairing", bargap=0.32,
        )
        plot(fig, key="stiffness", height=420)

    with s2c:
        theme.card(st, "Why 400,000 is bad news, not good news", f"""
        <p>The model treats the structure as balls joined by springs. Four of
        the six pairings top out around {STIFFNESS_NORMAL_SCALE:.0f} — normal.
        But Peptide_07 on OmpA hits 12,000, and on OmpW, 400,000.</p>
        <p>That isn't a super-strong bond. Spring strength shoots up as two
        atoms get closer, so a handful of enormous values means a few atoms
        have been placed almost <b>on top of each other</b> — closer than
        {STERIC_CLASH_DISTANCE:.1f} ångström, which is physically impossible.
        The rest of the map is perfectly normal, which confirms it's a few bad
        contacts rather than a genuinely rigid structure.</p>
        <p>In other words the docking software produced an impossible pose. And
        catching that is exactly what this step is <i>for</i> — better to find
        it now than after weeks of simulation.</p>
        """)

    # ---- Flexibility profile
    theme.rule(st, "Which parts bend and which stay put")

    complex_names = ndf["complex"].tolist()
    default_complex = ("P3 × OmpW" if "P3 × OmpW" in complex_names
                       else complex_names[0])
    sel_complex = st.selectbox("Pairing", complex_names,
                               index=complex_names.index(default_complex),
                               key="tab4_complex")
    row = ndf[ndf["complex"] == sel_complex].iloc[0]

    pl, pr = st.columns([1.6, 1], gap="large")
    with pl:
        uploaded = st.file_uploader(
            "Have the real iMODS output? Drop it here (optional)",
            type=["txt", "dat", "csv"],
            help="One or two columns of numbers: position and bendiness.",
        )

        measured = None
        if uploaded is not None:
            try:
                cols, vals = hf.parse_profile_file(uploaded.getvalue())
                measured = (np.asarray(cols, dtype=float),
                            np.asarray(vals, dtype=float))
                st.success(f"Read {len(vals):,} positions from {uploaded.name}.",
                           icon="✅")
            except Exception as exc:
                st.error(f"Couldn't read that file: {exc}")

        if measured is not None:
            idx, defo = measured
            source_note = "Real measured data"
        else:
            # ---- SCHEMATIC ONLY
            # Drawn from the paper's written description of each profile so the
            # shape can be discussed. NOT iMODS output; labelled as such on the
            # chart and in a warning below.
            rng = np.random.default_rng(abs(hash(sel_complex)) % (2**31))
            n_receptor, n_peptide = 182, 23
            idx = np.arange(1, n_receptor + n_peptide + 1)
            defo = 0.18 + 0.05 * rng.random(len(idx))
            loops = 0.42 * np.abs(np.sin(idx / 9.0)) ** 6
            defo[:n_receptor] += loops[:n_receptor]

            tail = np.arange(n_peptide)
            if row["peptide"] == "Peptide_03" and row["target"] == "OmpW":
                centre = n_peptide / 2
                defo[n_receptor:] = 0.20 + 0.55 * ((tail - centre) / centre) ** 2
            elif row["peptide"] == "Peptide_07" and row["target"] == "LptDE":
                defo[n_receptor:] = 0.25 + 0.72 * (tail / max(1, n_peptide - 1)) ** 2.4
            elif row["steric_flag"]:
                defo[n_receptor:] = 0.21 + 0.09 * rng.random(n_peptide)
            else:
                defo[n_receptor:] = 0.22 + 0.30 * (
                    np.exp(-tail / 2.6) + np.exp(-(n_peptide - 1 - tail) / 2.6)
                )
            defo = np.clip(defo, 0.0, 1.05)
            source_note = "Sketch — not real measurements"

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=idx, y=defo, mode="lines", name="How much each part bends",
            line=dict(color=theme.CATEGORICAL[0], width=2),
            fill="tozeroy", fillcolor="rgba(216,27,116,0.10)",
            hovertemplate="Position %{x}<br>bendiness %{y:.3f}<extra></extra>",
        ))
        if measured is None:
            fig.add_vline(x=182, line=dict(color=theme.ROSE_DEEP, dash="dash", width=2),
                          annotation_text="peptide starts here",
                          annotation_position="top left")
        fig.update_layout(title=f"{sel_complex} — {source_note}",
                          xaxis_title="Position along the chain",
                          yaxis_title="Bendiness", showlegend=False)
        plot(fig, key="deformability", height=420)

        if measured is None:
            st.warning(
                "This line is a **sketch** drawn from the written description "
                "in the paper — the shape is right but the numbers aren't real "
                "measurements. Upload the iMODS file above to show the real one.",
                icon="📐",
            )

    with pr:
        st.metric("Stiffness", f"{row['lambda_1']:.2e}")
        st.metric("Main wobble", f"{row['mode1_variance']*100:.0f}% of movement")
        st.metric("Top three wobbles", f"{row['cum3_variance']*100:.0f}% of movement")
        st.metric("Strongest spring", f"{row['stiffness_max']:,.0f}",
                  delta="atoms overlapping" if row["steric_flag"] else "normal",
                  delta_color="inverse" if row["steric_flag"] else "normal")
        theme.card(st, "What was seen here", f"<p>{row['mobility']}</p>")
        st.caption(
            "Stiff sections are the barrel walls holding the protein's shape; "
            "bendy ones are the loops dangling off the ends. For Peptide_03, "
            "the predicted pattern closely matches real X-ray measurements of "
            "LptDE and OmpW deposited in the PDB — good evidence the model is "
            "behaving like the real molecule."
        )

    theme.rule(st, "All six pairings")
    tbl = ndf[["complex", "peptide", "target", "pdb", "lambda_1",
               "mode1_variance", "cum3_variance", "stiffness_max",
               "steric_flag"]].copy()
    tbl.columns = ["Pairing", "Peptide", "Protein", "PDB code", "Stiffness",
                   "Main wobble", "Top three", "Strongest spring",
                   "Atoms overlapping?"]
    tbl["Stiffness"] = tbl["Stiffness"].map(lambda v: f"{v:.2e}")
    tbl["Main wobble"] = tbl["Main wobble"].map(lambda v: f"{v*100:.0f}%")
    tbl["Top three"] = tbl["Top three"].map(lambda v: f"{v*100:.0f}%")
    tbl["Strongest spring"] = tbl["Strongest spring"].map(lambda v: f"{v:,.0f}")
    tbl["Atoms overlapping?"] = tbl["Atoms overlapping?"].map(
        lambda b: "Yes — problem" if b else "No")
    st.dataframe(tbl, use_container_width=True, hide_index=True)

    st.success(
        "**The pipeline did what it was built to do.** Starting from more "
        "than a trillion trillion trillion possible sequences, it produced two "
        "candidates that hold onto the bacterium's surface proteins as firmly "
        "as peptides already known to work, keep their shape when shaken, and "
        "carry low predicted risk to human cells.\n\n"
        "**Peptide_03 is the stronger of the two.** It sits properly on all "
        "three targets, grips in the middle while its ends stay free, and its "
        "predicted flexing matches real X-ray measurements of the same "
        "proteins. Peptide_07 performs well on LptDE. These are the molecules "
        "a laboratory would pick up and test next.",
        icon="🌸",
    )

# --- Footer ---

theme.gap(st, 20)

theme.site_footer(
    st,
    columns=[
        ("The project", f"""
         <p><b>{PROJECT['acronym']}</b></p>
         <p>{PROJECT['title']}</p>
         <p>{PROJECT['category']}, {PROJECT['year']}</p>
        """),
        ("Researchers", "<ul>" + "".join(
            f"<li>{a}</li>" for a in PROJECT["authors"]) + "</ul>"),
        ("Guidance", f"""
         <p><b>{PROJECT['adviser']}</b><br>Research adviser</p>
         <p><b>{PROJECT['qualified_scientist']}</b><br>Qualified scientist</p>
         <p>{PROJECT['institution']}</p>
        """),
        ("Data sources", "<ul>" + "".join(
            f'<li><a href="{u}" target="_blank" rel="noopener">{n} \u2197</a></li>'
            for n, u in REPOSITORIES.items()) + "</ul>"),
    ],
    tagline="NURTURING THE BREAKTHROUGH",
    colophon=(
        f"{PROJECT['acronym']} &middot; {PROJECT['institution']} "
        f"&middot; {PROJECT['year']}"
    ),
)
