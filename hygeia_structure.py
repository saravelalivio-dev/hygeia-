"""3-D viewer for docked peptide-protein complexes, using 3Dmol.js."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List

# Biggest file we'll try to draw in the browser.
MAX_INLINE_BYTES = 8 * 1024 * 1024

STRUCTURE_DIR = "structures"
STRUCTURE_EXTENSIONS = (".pdb", ".ent", ".cif", ".mmcif")

# 3Dmol.js is loaded from the web, so nothing needs installing.
THREEDMOL_CDN = "https://3Dmol.org/build/3Dmol-min.js"

# PDB codes for the three target proteins.
RECEPTOR_IDS: Dict[str, str] = {
    "OmpA": "2K0L",
    "OmpW": "8QXP",
    "LptDE": "5IV9",
}

# Peptide in pink, protein in blue.
PEPTIDE_COLOUR = "#D81B74"
RECEPTOR_COLOUR = "#7FB3C8"
SURFACE_COLOUR = "#FFD0E4"
BACKGROUND = "#FFF5F9"

STYLE_CHOICES = {
    "Cartoon": "cartoon",
    "Cartoon + sticks": "cartoon_stick",
    "Sticks": "stick",
    "Spheres": "sphere",
    "Surface": "surface",
}


@dataclass
class Structure:
    """A structure file we can display."""

    label: str
    text: str
    fmt: str
    source: str          # "bundled" | "uploaded" | "rcsb"
    path: str | None = None


# --- Discovery ---

def list_bundled(folder: str = STRUCTURE_DIR) -> List[str]:
    """List the structure files in the structures/ folder."""
    if not os.path.isdir(folder):
        return []
    return sorted(
        f for f in os.listdir(folder)
        if f.lower().endswith(STRUCTURE_EXTENSIONS)
    )


def load_bundled(filename: str, folder: str = STRUCTURE_DIR) -> Structure:
    """Load a structure file. basename() stops paths like ../secrets."""
    safe = os.path.basename(filename)             # no "../" traversal
    path = os.path.join(folder, safe)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{safe} is not in the {folder}/ folder.")
    if os.path.getsize(path) > MAX_INLINE_BYTES:
        raise ValueError(
            f"{safe} is larger than {MAX_INLINE_BYTES // (1024*1024)} MB, "
            "too big to render in the browser."
        )
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    return Structure(label=safe, text=text, fmt=_format_of(safe),
                     source="bundled", path=path)


def _format_of(name: str) -> str:
    """Work out the file format from the name."""
    return "mmcif" if name.lower().endswith((".cif", ".mmcif")) else "pdb"


def summarise(structure: Structure) -> Dict[str, object]:
    """Count atoms, chains and residues.

    HPEPDOCK saves the protein and the peptide as separate chains, so this
    also guesses which chain is the peptide (the shorter one).
    """
    atoms = 0
    chains: Dict[str, int] = {}
    residues: Dict[str, set] = {}

    for line in structure.text.splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        atoms += 1
        if len(line) > 21:
            ch = line[21].strip() or "?"
            chains[ch] = chains.get(ch, 0) + 1
            if len(line) > 26:
                residues.setdefault(ch, set()).add(line[22:27].strip())

    return {
        "atoms": atoms,
        "chains": chains,
        "residues": {c: len(r) for c, r in residues.items()},
        # The shortest chain is almost always the docked peptide: 18-28
        # residues against a multi-hundred-residue barrel.
        "likely_peptide": (min(residues, key=lambda c: len(residues[c]))
                           if residues else None),
    }


# --- Viewer Html ---

def _js_string(value: str) -> str:
    """Turn a Python string into a safe JavaScript string.

    json.dumps escapes quotes and newlines but not "/", so a file containing
    "</script>" would end the script tag early and break the page.
    """
    return json.dumps(value).replace("</", "<\\/")


def _style_js(style_key: str, peptide_chain: str | None) -> str:
    """Build the JavaScript that styles the model."""
    mode = STYLE_CHOICES.get(style_key, "cartoon")
    pep = _js_string(peptide_chain) if peptide_chain else "null"

    base = {
        "cartoon": f'{{cartoon: {{color: "{RECEPTOR_COLOUR}", opacity: 0.92}}}}',
        "cartoon_stick": f'{{cartoon: {{color: "{RECEPTOR_COLOUR}", opacity: 0.85}}}}',
        "stick": f'{{stick: {{radius: 0.14, color: "{RECEPTOR_COLOUR}"}}}}',
        "sphere": f'{{sphere: {{scale: 0.26, color: "{RECEPTOR_COLOUR}"}}}}',
        "surface": f'{{cartoon: {{color: "{RECEPTOR_COLOUR}", opacity: 0.9}}}}',
    }[mode]

    peptide = {
        "cartoon": f'{{cartoon: {{color: "{PEPTIDE_COLOUR}"}}, '
                   f'stick: {{radius: 0.17, color: "{PEPTIDE_COLOUR}"}}}}',
        "cartoon_stick": f'{{cartoon: {{color: "{PEPTIDE_COLOUR}"}}, '
                         f'stick: {{radius: 0.19, color: "{PEPTIDE_COLOUR}"}}}}',
        "stick": f'{{stick: {{radius: 0.22, color: "{PEPTIDE_COLOUR}"}}}}',
        "sphere": f'{{sphere: {{scale: 0.34, color: "{PEPTIDE_COLOUR}"}}}}',
        "surface": f'{{cartoon: {{color: "{PEPTIDE_COLOUR}"}}, '
                   f'stick: {{radius: 0.2, color: "{PEPTIDE_COLOUR}"}}}}',
    }[mode]

    surface = ""
    if mode == "surface":
        surface = f"""
        try {{
            var sel = {pep} ? {{chain: {pep}, invert: true}} : {{}};
            viewer.addSurface($3Dmol.SurfaceType.VDW,
                {{opacity: 0.62, color: "{SURFACE_COLOUR}"}}, sel);
        }} catch (e) {{ console.warn("surface failed", e); }}
        """

    return f"""
        viewer.setStyle({{}}, {base});
        var pepChain = {pep};
        if (pepChain) {{
            viewer.setStyle({{chain: pepChain}}, {peptide});
        }}
        {surface}
    """


def viewer_html(structure: Structure, style: str = "Cartoon",
                peptide_chain: str | None = None, spin: bool = False,
                height: int = 520) -> str:
    """Build the HTML that draws one structure."""
    if len(structure.text.encode("utf-8")) > MAX_INLINE_BYTES:
        raise ValueError("Structure too large to display.")

    payload = _js_string(structure.text)
    fmt = _js_string(structure.fmt)

    return f"""
<div style="position:relative;width:100%;height:{height}px;
            border-radius:18px;overflow:hidden;
            border:1px solid #FFD0E4;background:{BACKGROUND};">
  <div id="hygeia-viewport" style="width:100%;height:100%;position:relative;"></div>
  <div id="hygeia-status" style="position:absolute;top:12px;left:14px;
       font:600 12px/1.4 system-ui,sans-serif;color:#A4134E;
       background:rgba(255,255,255,0.86);padding:5px 11px;border-radius:999px;
       border:1px solid #FFD0E4;">loading…</div>
</div>
<script src="{THREEDMOL_CDN}"></script>
<script>
(function () {{
    var status = document.getElementById("hygeia-status");

    function start() {{
        if (typeof $3Dmol === "undefined") {{
            status.textContent = "3Dmol.js did not load — check the connection";
            return;
        }}
        try {{
            var el = document.getElementById("hygeia-viewport");
            var viewer = $3Dmol.createViewer(el, {{backgroundColor: "{BACKGROUND}"}});
            viewer.addModel({payload}, {fmt});
            {_style_js(style, peptide_chain)}
            viewer.zoomTo();
            viewer.render();
            viewer.zoom(1.15, 800);
            {"viewer.spin('y', 0.6);" if spin else ""}
            status.textContent = "drag to rotate · scroll to zoom";
            setTimeout(function () {{ status.style.opacity = "0.45"; }}, 2600);
        }} catch (err) {{
            status.textContent = "Could not render: " + err.message;
            console.error(err);
        }}
    }}

    // 3Dmol.js may not have finished loading yet.
    if (typeof $3Dmol !== "undefined") {{ start(); }}
    else {{
        var tries = 0;
        var t = setInterval(function () {{
            if (typeof $3Dmol !== "undefined") {{ clearInterval(t); start(); }}
            else if (++tries > 100) {{
                clearInterval(t);
                status.textContent = "3Dmol.js timed out — check the connection";
            }}
        }}, 100);
    }}
}})();
</script>
"""


def rcsb_viewer_html(pdb_id: str, style: str = "Cartoon",
                     spin: bool = False, height: int = 520) -> str:
    """Build the HTML that downloads a structure from the PDB and draws it.

    This is the protein on its own. Docked complexes only exist in our own
    HPEPDOCK output files.
    """
    safe_id = "".join(ch for ch in pdb_id if ch.isalnum())[:8].upper()
    mode = STYLE_CHOICES.get(style, "cartoon")
    style_js = {
        "cartoon": f'{{cartoon: {{color: "spectrum"}}}}',
        "cartoon_stick": f'{{cartoon: {{color: "spectrum"}}, stick: {{radius: 0.1}}}}',
        "stick": '{stick: {radius: 0.14}}',
        "sphere": '{sphere: {scale: 0.26}}',
        "surface": f'{{cartoon: {{color: "spectrum"}}}}',
    }[mode]

    return f"""
<div style="position:relative;width:100%;height:{height}px;
            border-radius:18px;overflow:hidden;
            border:1px solid #FFD0E4;background:{BACKGROUND};">
  <div id="hygeia-rcsb" style="width:100%;height:100%;position:relative;"></div>
  <div id="hygeia-rcsb-status" style="position:absolute;top:12px;left:14px;
       font:600 12px/1.4 system-ui,sans-serif;color:#A4134E;
       background:rgba(255,255,255,0.86);padding:5px 11px;border-radius:999px;
       border:1px solid #FFD0E4;">fetching {safe_id} from RCSB…</div>
</div>
<script src="{THREEDMOL_CDN}"></script>
<script>
(function () {{
    var status = document.getElementById("hygeia-rcsb-status");

    function start() {{
        if (typeof $3Dmol === "undefined") {{
            status.textContent = "3Dmol.js did not load";
            return;
        }}
        var el = document.getElementById("hygeia-rcsb");
        var viewer = $3Dmol.createViewer(el, {{backgroundColor: "{BACKGROUND}"}});
        $3Dmol.download("pdb:{safe_id}", viewer, {{}}, function () {{
            try {{
                viewer.setStyle({{}}, {style_js});
                viewer.zoomTo();
                viewer.render();
                {"viewer.spin('y', 0.6);" if spin else ""}
                status.textContent = "{safe_id} · receptor only, no peptide bound";
                setTimeout(function () {{ status.style.opacity = "0.45"; }}, 3200);
            }} catch (err) {{
                status.textContent = "Render failed: " + err.message;
            }}
        }});
        setTimeout(function () {{
            if (status.textContent.indexOf("fetching") === 0) {{
                status.textContent = "RCSB did not respond — try again";
            }}
        }}, 14000);
    }}

    if (typeof $3Dmol !== "undefined") {{ start(); }}
    else {{
        var tries = 0;
        var t = setInterval(function () {{
            if (typeof $3Dmol !== "undefined") {{ clearInterval(t); start(); }}
            else if (++tries > 100) {{ clearInterval(t); status.textContent = "3Dmol.js timed out"; }}
        }}, 100);
    }}
}})();
</script>
"""
