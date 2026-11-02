"""
Physicochemical properties of a peptide sequence.

Each function follows the equation given in the paper. Written out here rather
than using modlAMP so the numbers match exactly and there is one less package
to install.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

# --- Constants ---

CANONICAL_AA = "ACDEFGHIKLMNPQRSTVWY"

# Kyte-Doolittle scale. Positive means water-repelling.
KYTE_DOOLITTLE: Dict[str, float] = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

# Eisenberg normalised scale (Eisenberg et al., 1984), used for mu_H.
# Some packages use the un-normalised version instead, which gives moments
# about 1.6x larger. Same ranking, different absolute numbers.
EISENBERG: Dict[str, float] = {
    "A": 0.25, "R": -1.76, "N": -0.64, "D": -0.72, "C": 0.04,
    "Q": -0.69, "E": -0.62, "G": 0.16, "H": -0.40, "I": 0.73,
    "L": 0.53, "K": -1.10, "M": 0.26, "F": 0.61, "P": -0.07,
    "S": -0.26, "T": -0.18, "W": 0.37, "Y": 0.02, "V": 0.54,
}

# pKa values (EMBOSS set) for the groups that gain or lose charge.
PKA_BASIC = {"K": 10.00, "R": 12.00, "H": 6.00}

# D and E only. Cysteine and tyrosine are weakly acidic and barely change the
# charge at pH 7, but they shift pI by a whole unit, and leaving them out is
# what reproduces both columns of Table 6.
PKA_ACIDIC = {"D": 4.05, "E": 4.45}
PKA_ACIDIC_EXTENDED = {"D": 4.05, "E": 4.45, "C": 9.00, "Y": 10.00}
PKA_N_TERM = 9.60   # alpha-amino group, basic
PKA_C_TERM = 3.65   # alpha-carboxyl group, acidic

# Degrees of turn per residue in an alpha-helix.
HELIX_ANGLE_DEG = 100.0

# Converts our mu_H values to the scale used in Table 6, for display only.
# Measured as 1.572 +/- 0.034 across all nine peptides.
MOMENT_SCALE_RATIO = 1.572

# Aliphatic index weights, relative to alanine.
AI_COEFF_VAL = 2.9   # a
AI_COEFF_ILE_LEU = 3.9  # b


# --- Validation ---

def clean_sequence(seq: str) -> str:
    """Upper-case, strip whitespace/digits/FASTA headers, keep canonical residues."""
    if not seq:
        return ""
    lines = [ln for ln in seq.splitlines() if not ln.strip().startswith(">")]
    merged = "".join(lines).upper()
    return "".join(ch for ch in merged if ch in CANONICAL_AA)


def validate_sequence(seq: str, min_len: int = 10, max_len: int = 30) -> Tuple[bool, str]:
    """Check the length is within the 10-30 residue design window.

    Outside that range the model has no training data to go on.
    """
    cleaned = clean_sequence(seq)
    if not cleaned:
        return False, "No canonical amino acid residues found."
    if len(cleaned) < min_len:
        return False, f"Length {len(cleaned)} is below the {min_len}-residue design floor."
    if len(cleaned) > max_len:
        return False, f"Length {len(cleaned)} exceeds the {max_len}-residue design ceiling."
    return True, "Sequence accepted into the design window."


# --- Net Charge ---

def net_charge(seq: str, ph: float = 7.0, c_term_amide: bool = True,
               acidic: Dict[str, float] | None = None) -> float:
    """Net charge at a given pH, using Henderson-Hasselbalch.

        z = sum_basic  1 / (1 + 10^(pH - pKa))
          - sum_acidic 1 / (1 + 10^(pKa - pH))

    Each group is counted as a fraction rather than a whole charge, which is
    why we get values like +4.975 instead of +5.

    c_term_amide=True treats the C-terminus as amidated (standard for
    synthetic AMPs, and what matches our published charges). Set False for a
    free acid.
    """
    s = clean_sequence(seq)
    if not s:
        return 0.0

    positive = 1.0 / (1.0 + 10 ** (ph - PKA_N_TERM))      # free alpha-amino terminus
    for residue, pka in PKA_BASIC.items():
        positive += s.count(residue) / (1.0 + 10 ** (ph - pka))

    negative = 0.0
    if not c_term_amide:                                  # free alpha-carboxyl
        negative += 1.0 / (1.0 + 10 ** (PKA_C_TERM - ph))
    for residue, pka in (acidic or PKA_ACIDIC).items():
        negative += s.count(residue) / (1.0 + 10 ** (pka - ph))

    return positive - negative


def isoelectric_point(seq: str, lo: float = 0.0, hi: float = 14.0,
                      tol: float = 1e-4, c_term_amide: bool = False) -> float:
    """The pH where net charge is zero, found by bisection.

    There is no way to solve the equation directly, so we keep halving the
    range. Charge always falls as pH rises, so this always converges.

    Unlike net_charge, this assumes a free C-terminus, which is what every
    standard pI calculator does. With an amidated one, a strongly positive
    peptide never reaches zero charge at all.
    """
    s = clean_sequence(seq)
    if not s:
        return 0.0
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if net_charge(s, mid, c_term_amide=c_term_amide) > 0:
            lo = mid        # still net positive, move right
        else:
            hi = mid        # already net negative, move left
    return 0.5 * (lo + hi)


# --- Hydrophobic Moment ---

def hydrophobic_moment(seq: str, window: int | None = None,
                       angle_deg: float = HELIX_ANGLE_DEG,
                       scale: Dict[str, float] | None = None) -> float:
    """Eisenberg hydrophobic moment.

        mu_H = (1/N) * sqrt( (sum H_n sin d_n)^2 + (sum H_n cos d_n)^2 )

    Residues are placed every 100 degrees around a helical wheel and weighted
    by hydrophobicity. The total is only large when the water-repelling
    residues end up on the same side of the helix, which is what a peptide
    needs to push into a membrane.

    window defaults to the whole sequence, as in the paper. Pass 11 to scan
    sliding windows and take the best one instead.
    """
    s = clean_sequence(seq)
    if not s:
        return 0.0
    scale = scale or EISENBERG
    win = len(s) if window is None else min(window, len(s))
    radians = math.radians(angle_deg)

    best = 0.0
    for start in range(0, len(s) - win + 1):
        sin_sum = cos_sum = 0.0
        for n, residue in enumerate(s[start:start + win]):
            h = scale.get(residue, 0.0)
            sin_sum += h * math.sin(n * radians)
            cos_sum += h * math.cos(n * radians)
        # Divide by length so different window sizes compare fairly.
        best = max(best, math.hypot(sin_sum, cos_sum) / win)
    return best


# --- Gravy And Aliphatic Index ---

def gravy(seq: str) -> float:
    """GRAVY = average Kyte-Doolittle value over the sequence.

    How water-repelling the peptide is overall, ignoring where those residues
    sit. mu_H covers the arrangement.
    """
    s = clean_sequence(seq)
    if not s:
        return 0.0
    return sum(KYTE_DOOLITTLE.get(r, 0.0) for r in s) / len(s)


def aliphatic_index(seq: str) -> float:
    """Ikai aliphatic index.

        AI = X_Ala + 2.9 * X_Val + 3.9 * (X_Ile + X_Leu)

    X are mole percentages. A high value means the peptide holds its shape
    when warm, but also that it clumps more easily, so it cuts both ways.
    """
    s = clean_sequence(seq)
    if not s:
        return 0.0
    L = len(s)
    mol_pct = lambda r: 100.0 * s.count(r) / L  # noqa: E731
    return (
        mol_pct("A")
        + AI_COEFF_VAL * mol_pct("V")
        + AI_COEFF_ILE_LEU * (mol_pct("I") + mol_pct("L"))
    )


def aggregation_band(ai: float) -> str:
    """Which clumping-risk band an aliphatic index falls into."""
    if ai < 50.0:
        return "Low aggregation risk"
    if ai < 80.0:
        return "Normal behaviour"
    return "High aggregation risk"


# --- Aggregate Descriptor Vector ---

# Feature order the model expects. Don't reorder: the model goes by position,
# so swapping two entries would quietly break every prediction.
FEATURE_ORDER: List[str] = [
    "length", "charge", "mu_h", "gravy", "ai", "pI",
    "frac_cationic", "frac_hydrophobic", "frac_aromatic", "frac_gly_pro",
    "frac_cys", "frac_anionic",
]

HYDROPHOBIC_SET = set("AILMFVWY")
AROMATIC_SET = set("FWY")
CATIONIC_SET = set("KRH")
ANIONIC_SET = set("DE")


def describe(seq: str, ph: float = 7.0,
             c_term_amide: bool = True) -> Dict[str, float]:
    """All twelve physicochemical features for one sequence.

    The full model also used 160 PepFoundry embedding features, which need the
    embedding model to compute, so they aren't recalculated here.
    """
    s = clean_sequence(seq)
    if not s:
        return {k: 0.0 for k in FEATURE_ORDER}
    L = len(s)
    frac = lambda members: sum(1 for r in s if r in members) / L  # noqa: E731

    return {
        "length": float(L),
        "charge": net_charge(s, ph, c_term_amide=c_term_amide),
        "mu_h": hydrophobic_moment(s),
        "gravy": gravy(s),
        "ai": aliphatic_index(s),
        # pI keeps the free-carboxyl convention regardless of the charge
        # convention above - see isoelectric_point's docstring.
        "pI": isoelectric_point(s),
        "frac_cationic": frac(CATIONIC_SET),
        "frac_hydrophobic": frac(HYDROPHOBIC_SET),
        "frac_aromatic": frac(AROMATIC_SET),
        "frac_gly_pro": frac(set("GP")),
        "frac_cys": frac(set("C")),
        "frac_anionic": frac(ANIONIC_SET),
    }


def feature_vector(seq: str, ph: float = 7.0) -> List[float]:
    """Descriptor dict flattened into FEATURE_ORDER, ready for the classifier."""
    d = describe(seq, ph)
    return [d[k] for k in FEATURE_ORDER]


# --- Reading iMODS output files ---

def parse_profile_file(raw: bytes | str) -> Tuple[List[float], List[float]]:
    """Read an iMODS export into (index, value) lists.

    Handles spaces, tabs or commas, with or without a header row. Written by
    hand because pandas' delimiter sniffer splits a single column of decimals
    on the decimal point.
    """
    text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw

    idx: List[float] = []
    val: List[float] = []
    for line in text.splitlines():
        line = line.split("#")[0].strip()
        if not line:
            continue
        tokens = [t for t in line.replace(",", " ").replace(";", " ").split() if t]
        numbers: List[float] = []
        for tok in tokens[:3]:
            try:
                numbers.append(float(tok))
            except ValueError:
                break           # a header row, or a trailing text label
        if not numbers:
            continue            # header or comment, skip without complaint
        if len(numbers) == 1:
            idx.append(len(val) + 1.0)
            val.append(numbers[0])
        else:
            idx.append(numbers[0])
            val.append(numbers[1])

    if not val:
        raise ValueError("No numeric rows found — expected one or two columns "
                         "of numbers, optionally with a header or comments.")
    return idx, val


# Labels shown in the app.
FEATURE_LABELS = {
    "length": "Sequence length",
    "charge": "Net charge (z)",
    "mu_h": "Hydrophobic moment",
    "gravy": "GRAVY",
    "ai": "Aliphatic index",
    "pI": "Isoelectric point",
    "frac_cationic": "Cationic fraction",
    "frac_hydrophobic": "Hydrophobic fraction",
    "frac_aromatic": "Aromatic fraction",
    "frac_gly_pro": "Gly/Pro fraction",
    "frac_cys": "Cysteine fraction",
    "frac_anionic": "Anionic fraction",
}
