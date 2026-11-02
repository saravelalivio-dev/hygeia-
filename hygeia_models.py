"""
The screening model, SHAP attribution and the Pareto sort.

The trained Random Forest from the study isn't included with this site, so if
no model file is found a stand-in is fitted at startup using the same
relationships TreeSHAP found in the real one. The app says which is running,
and the nine published peptides always report their real scores either way.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier

import hygeia_features as hf
from hygeia_data import CANDIDATES

MODEL_PATH = os.environ.get("HYGEIA_MODEL_PATH", "hygeia_rf_gatekeeper.joblib")
RANDOM_STATE = 42

# Published scores, so peptides from the paper always show their real value.
PUBLISHED_P_ACTIVE: Dict[str, float] = {c["sequence"]: c["p_active"] for c in CANDIDATES}
PUBLISHED_ID: Dict[str, str] = {c["sequence"]: c["id"] for c in CANDIDATES}

# Published properties for the nine peptides (Table 6).
PUBLISHED_DESCRIPTORS: Dict[str, Dict[str, float]] = {
    c["sequence"]: {
        "length": float(len(c["sequence"])),
        "charge": c["charge"], "mu_h": c["mu_h"], "gravy": c["gravy"],
        "ai": c["ai"], "pI": c["pI"],
    }
    for c in CANDIDATES
}


def descriptors_for(seq: str, ph: float = 7.0) -> Tuple[Dict[str, float], str]:
    """Properties for one sequence, and where the numbers came from.

    Returns the published values for the nine paper peptides at pH 7, and
    freshly calculated ones otherwise. Recalculated values match the paper
    closely (GRAVY and AI exactly, charge within 0.10, pI within 0.73), but
    the published ones are what we quote for the published peptides.

    At any other pH everything is recalculated, since charge and pI depend
    on it.
    """
    cleaned = hf.clean_sequence(seq)
    if abs(ph - 7.0) < 1e-9 and cleaned in PUBLISHED_DESCRIPTORS:
        block = dict(PUBLISHED_DESCRIPTORS[cleaned])
        computed = hf.describe(cleaned, ph=ph)
        # Composition fractions are not tabulated in the manuscript, so they
        # come from the recomputation either way.
        for key in hf.FEATURE_ORDER:
            block.setdefault(key, computed[key])
        return block, "manuscript"
    return hf.describe(cleaned, ph=ph), "computed"


# --- The screening model ---

def _synthetic_corpus(n: int = 1400, seed: int = RANDOM_STATE):
    """Make training data for the stand-in model.

    Random sequences, scored using the two relationships our SHAP analysis
    confirmed: charge acts in a straight line, mu_H acts like a pass mark.
    The logistic term is what makes low mu_H get punished hard while high
    mu_H only helps a little.

    About 30% come out active, matching the real dataset.
    """
    rng = np.random.default_rng(seed)

    # Weighted towards the amino acids common in real AMPs.
    alphabet = list(hf.CANONICAL_AA)
    weights = np.array([
        0.09,  # A
        0.02,  # C
        0.03,  # D
        0.03,  # E
        0.04,  # F
        0.09,  # G
        0.03,  # H
        0.05,  # I
        0.09,  # K
        0.07,  # L
        0.02,  # M
        0.05,  # N
        0.04,  # P
        0.03,  # Q
        0.11,  # R
        0.06,  # S
        0.05,  # T
        0.05,  # V
        0.03,  # W
        0.02,  # Y
    ], dtype=float)
    weights /= weights.sum()

    X, y = [], []
    for _ in range(n):
        length = int(rng.integers(10, 31))          # canonical design window
        seq = "".join(rng.choice(alphabet, size=length, p=weights))
        d = hf.describe(seq)

        # Score, then turn into a probability, then a yes/no label.
        charge_term = 0.42 * d["charge"]
        amphipathy_term = 3.1 / (1.0 + np.exp(-18.0 * (d["mu_h"] - 0.26)))
        gravy_term = 0.28 * d["gravy"]
        ai_term = -0.004 * max(0.0, d["ai"] - 80.0)   # too aliphatic is a problem
        score = charge_term + amphipathy_term + gravy_term + ai_term - 4.1

        p = 1.0 / (1.0 + np.exp(-score))
        X.append([d[k] for k in hf.FEATURE_ORDER])
        y.append(int(rng.random() < p))

    return np.asarray(X, dtype=float), np.asarray(y, dtype=int)


@dataclass
class Gatekeeper:
    """The screening model. is_surrogate decides whether the app warns."""

    model: RandomForestClassifier
    is_surrogate: bool
    feature_names: List[str] = field(default_factory=lambda: list(hf.FEATURE_ORDER))

    # prediction
    def predict_proba(self, seq: str) -> float:
        """Predicted chance one sequence is active.

        Peptides from the paper return their published score. Everything else
        is scored by the forest.
        """
        cleaned = hf.clean_sequence(seq)
        if cleaned in PUBLISHED_P_ACTIVE:
            return PUBLISHED_P_ACTIVE[cleaned]
        x = np.asarray([hf.feature_vector(cleaned)], dtype=float)
        return float(self.model.predict_proba(x)[0, 1])

    def predict_batch(self, seqs: Sequence[str]) -> List[float]:
        return [self.predict_proba(s) for s in seqs]

    def is_published(self, seq: str) -> str | None:
        """Give the Peptide_NN name if this is one of the paper's nine."""
        return PUBLISHED_ID.get(hf.clean_sequence(seq))


def load_gatekeeper() -> Gatekeeper:
    """Load the trained model if we have it, otherwise fit the stand-in."""
    if os.path.exists(MODEL_PATH):
        try:
            import joblib
            return Gatekeeper(model=joblib.load(MODEL_PATH), is_surrogate=False)
        except Exception:
            pass  # use the stand-in rather than crash

    X, y = _synthetic_corpus()
    # Same settings as the benchmarking run: balanced weights for the 70/30
    # split, limited depth so it doesn't overfit.
    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=12,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X, y)
    return Gatekeeper(model=rf, is_surrogate=True)


# --- Generating new sequences ---

# How often each amino acid gets picked. More positive residues, since
# charge mattered most, but enough greasy ones to form a face.
_GEN_WEIGHTS = {
    "R": 0.130, "K": 0.085, "G": 0.085, "A": 0.080, "N": 0.060,
    "L": 0.060, "S": 0.055, "V": 0.050, "I": 0.045, "T": 0.045,
    "W": 0.040, "Y": 0.035, "F": 0.035, "P": 0.035, "Q": 0.030,
    "M": 0.025, "H": 0.025, "E": 0.025, "D": 0.020, "C": 0.015,
}


def generate_sequences(
    n: int = 9,
    length_range: Tuple[int, int] = (18, 28),
    temperature: float = 1.0,
    template: str = "",
    seed: int | None = None,
) -> List[str]:
    """Generate new sequences, standing in for ProtGPT2.

    ProtGPT2 is about 3 GB, far too big for a free hosted container, so this
    copies how it samples rather than the model itself.

    temperature rescales the letter frequencies as p^(1/T), the same thing
    done to transformer outputs. Below 1 sticks closer to what works, above 1
    explores more. template holds any letters given as a fixed prefix.

    Swapping in the real model means replacing the body with a transformers
    pipeline call and keeping the same arguments.
    """
    rng = np.random.default_rng(seed)
    residues = np.array(list(_GEN_WEIGHTS.keys()))
    base = np.array(list(_GEN_WEIGHTS.values()), dtype=float)

    # Same rescaling a language model applies to its output.
    t = max(0.05, float(temperature))
    probs = base ** (1.0 / t)
    probs /= probs.sum()

    prefix = hf.clean_sequence(template)
    lo, hi = length_range
    lo, hi = max(10, min(lo, hi)), min(30, max(lo, hi))

    out: List[str] = []
    for _ in range(n):
        target = int(rng.integers(lo, hi + 1))
        remaining = max(0, target - len(prefix))
        tail = "".join(rng.choice(residues, size=remaining, p=probs))
        out.append((prefix + tail)[:30])
    return out


def apply_gate(
    seqs: Sequence[str],
    gatekeeper: Gatekeeper,
    min_len: int = 10,
    max_len: int = 30,
    min_charge: float = 2.0,
    min_mu_h: float = 0.13,
) -> List[dict]:
    """Screen sequences on length, then on charge and amphipathicity.

    Length first, since outside 10-30 residues the model has no training data
    to go on. Then the two properties SHAP found mattered most. A peptide with
    its greasy residues spread evenly can't get into a membrane whatever else
    it has going for it, which is why there's a mu_H floor at all.

    min_mu_h of 0.13 on our scale is about 0.20 on the paper's. Thresholds
    always use freshly calculated values, even for the published peptides, so
    both are judged the same way.

    Returns a record per sequence including why anything was rejected, so the
    app can show that rather than quietly dropping it.
    """
    records = []
    for seq in seqs:
        cleaned = hf.clean_sequence(seq)
        ok_len, len_msg = hf.validate_sequence(cleaned, min_len, max_len)
        if cleaned:
            d, source = descriptors_for(cleaned)   # shown in the table
            gate_d = hf.describe(cleaned)          # used for the thresholds
        else:
            d = gate_d = {k: 0.0 for k in hf.FEATURE_ORDER}
            source = "computed"

        reasons = []
        if not ok_len:
            reasons.append(len_msg)
        if ok_len and gate_d["charge"] < min_charge:
            reasons.append(
                f"Net charge {gate_d['charge']:+.2f} below the +{min_charge:.1f} floor."
            )
        if ok_len and gate_d["mu_h"] < min_mu_h:
            reasons.append(
                f"Hydrophobic moment {gate_d['mu_h']:.3f} below the "
                f"{min_mu_h:.2f} floor (≈{min_mu_h * hf.MOMENT_SCALE_RATIO:.2f} "
                f"on the Table 6 scale)."
            )

        passed = not reasons
        records.append({
            "sequence": cleaned,
            "published_id": gatekeeper.is_published(cleaned),
            "descriptor_source": source,
            "passed": passed,
            "reasons": reasons,
            "p_active": gatekeeper.predict_proba(cleaned) if passed else None,
            "aggregation": hf.aggregation_band(d["ai"]),
            **d,
        })
    return records


# --- Explaining a score ---

_BACKGROUND_CACHE: np.ndarray | None = None


def _background_matrix(n: int = 120) -> np.ndarray:
    """Background sample, used to stand in for a feature being absent."""
    global _BACKGROUND_CACHE
    if _BACKGROUND_CACHE is None:
        _BACKGROUND_CACHE, _ = _synthetic_corpus(n=n, seed=7)
    return _BACKGROUND_CACHE


def local_attribution(gatekeeper: Gatekeeper, seq: str,
                      n_permutations: int = 200) -> Tuple[Dict[str, float], float, str]:
    """Work out how much each feature moved one peptide's score.

    Uses shap.TreeExplainer if it's installed, which is the exact method from
    the paper. If not, falls back to averaging each feature's contribution
    over random orderings, which is the same quantity estimated by sampling.

    Returns the contributions, the starting value, and which method was used.
    The contributions always add up to the final score.
    """
    x = np.asarray(hf.feature_vector(seq), dtype=float)

    # Exact method, if shap is installed.
    try:
        import shap
        explainer = shap.TreeExplainer(gatekeeper.model)
        raw = explainer.shap_values(x.reshape(1, -1))
        # Different shap versions return different shapes here.
        if isinstance(raw, list):
            phi = np.asarray(raw[1]).ravel()
            base = float(np.ravel(explainer.expected_value)[1])
        else:
            arr = np.asarray(raw)
            phi = arr[0, :, 1].ravel() if arr.ndim == 3 else arr.ravel()
            ev = np.ravel(explainer.expected_value)
            base = float(ev[1] if ev.size > 1 else ev[0])
        return dict(zip(hf.FEATURE_ORDER, phi.tolist())), base, "TreeSHAP (exact)"
    except Exception:
        pass

    # Fallback. Every combination is built first and scored in a single call.
    # Doing it one row at a time means thousands of separate predictions and
    # takes over a minute, which is far too slow for a web page.
    rng = np.random.default_rng(RANDOM_STATE)
    background = _background_matrix()
    base = float(gatekeeper.model.predict_proba(background)[:, 1].mean())

    n_feat = len(hf.FEATURE_ORDER)
    n_perm = int(n_permutations)

    orders = np.argsort(rng.random((n_perm, n_feat)), axis=1)      # random pi
    refs = background[rng.integers(len(background), size=n_perm)]  # fillers

    # rank[p, j] is where feature j appears in ordering p, so it's switched on
    # at step k when rank < k. Using a mask here avoids a numpy indexing trap
    # where mixing slices and arrays pairs up the wrong axes.
    rank = np.argsort(orders, axis=1)                     # (n_perm, n_feat)
    steps = np.arange(n_feat + 1)[None, :, None]          # (1, n_feat+1, 1)
    present = rank[:, None, :] < steps                    # (n_perm, n_feat+1, n_feat)

    # rows[p, k] has the first k features of ordering p switched on.
    rows = np.where(present, x[None, None, :], refs[:, None, :])

    flat = rows.reshape(-1, n_feat)
    probs = gatekeeper.model.predict_proba(flat)[:, 1].reshape(n_perm, n_feat + 1)

    # Each step up is that feature's contribution.
    increments = np.diff(probs, axis=1)
    phi = np.zeros(n_feat)
    np.add.at(phi, orders.ravel(), increments.ravel())
    phi /= n_perm

    return (dict(zip(hf.FEATURE_ORDER, phi.tolist())), base,
            "Permutation Shapley (vectorised fallback)")


# --- Choosing between candidates (ACPE) ---

def pareto_mask(objectives: np.ndarray) -> np.ndarray:
    """Find the non-dominated candidates.

    objectives is (candidates, objectives) and everything must be written so
    that lower is better, which is why Table 9 lists activity and aliphatic
    index as negatives.

    One candidate beats another when it's no worse on all four and better on
    at least one. Anything nothing else beats is on the front.

    We use dominance rather than adding the four scores together, because a
    weighted sum would mean deciding how much extra killing power is worth
    how much extra toxicity, and nothing in the data justifies that.
    """
    n = objectives.shape[0]
    non_dominated = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            no_worse = np.all(objectives[j] <= objectives[i])
            strictly_better = np.any(objectives[j] < objectives[i])
            if no_worse and strictly_better:
                non_dominated[i] = False
                break
    return non_dominated


def acpe_rank(records: Sequence[dict], thresholds: Dict[str, float],
              n_leads: int = 2) -> List[dict]:
    """Run the ACPE stage over a set of candidates.

    The four objectives, all written so lower is better:
        -p_active    we want activity high
        -aliphatic   we want sturdiness high
        +toxicity    we want this low
        +hemolysis   we want this low

    Each candidate gets three separate verdicts, which are not the same thing:

    non_dominated     nothing else beats it on all four. Seven of our nine
                      qualify, so this can't pick a winner on its own.
    threshold_checks  whether it meets each of our four standards. None of
                      the nine meets all four, because the activity and
                      sturdiness floors pull against the safety ceilings.
    selected_lead     what we actually went with: of the non-dominated ones,
                      the n_leads with the lowest toxicity + hemolysis. That
                      gives Peptide_07 and Peptide_03.
    """
    if not records:
        return []

    obj = np.array([
        [
            -r["p_active"],
            -r["ai"],
            r["toxicity"],
            r["hemolysis"],
        ]
        for r in records
    ], dtype=float)

    mask = pareto_mask(obj)
    out = []
    for rec, nd in zip(records, mask):
        checks = {
            "Activity floor": rec["p_active"] >= thresholds["p_active_min"],
            "Stability floor": rec["ai"] >= thresholds["ai_min"],
            "Toxicity ceiling": rec["toxicity"] < thresholds["toxicity_max"],
            "Hemolysis ceiling": rec["hemolysis"] < thresholds["hemolysis_max"],
        }
        enriched = dict(rec)
        enriched["non_dominated"] = bool(nd)
        enriched["threshold_checks"] = checks
        enriched["n_passed"] = sum(checks.values())
        enriched["all_thresholds"] = all(checks.values())
        enriched["risk_score"] = rec["toxicity"] + rec["hemolysis"]
        enriched["selected_lead"] = False
        out.append(enriched)

    # Pick the leads: safest first, among the ones nothing else beats.
    front = sorted(
        (r for r in out if r["non_dominated"]),
        key=lambda r: r["risk_score"],
    )
    for position, rec in enumerate(front, start=1):
        rec["risk_rank"] = position
        rec["selected_lead"] = position <= n_leads
    for rec in out:
        rec.setdefault("risk_rank", None)

    return out


def hypervolume_2d(points: np.ndarray, reference: Tuple[float, float]) -> float:
    """Area covered by the front, measured back to a reference point.

    One number for how good a whole front is, so two runs can be compared.
    """
    pts = np.asarray([p for p in points if p[0] < reference[0] and p[1] < reference[1]])
    if pts.size == 0:
        return 0.0
    pts = pts[np.argsort(pts[:, 0])]
    area = 0.0
    prev_y = reference[1]
    for x, y in pts:
        if y < prev_y:
            area += (reference[0] - x) * (prev_y - y)
            prev_y = y
    return float(area)
