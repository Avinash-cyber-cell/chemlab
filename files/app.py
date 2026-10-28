import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO

# RDKit
from rdkit import Chem
from rdkit.Chem import (
    Draw,
    AllChem,
    Descriptors,
    Crippen,
    Lipinski,
    rdMolDescriptors
)

# 3D Visualization
import py3Dmol
import streamlit.components.v1 as components

# PDF
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.colors import HexColor
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    Image,
    Preformatted
)
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
import tempfile
import os
import random


# ============================================================
# PAGE CONFIGURATION
# ============================================================

LOGO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "srm_logo.jpeg")

st.set_page_config(
    page_title="Cheminformatics Virtual Laboratory | SRM Tiruchirappalli",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS  (SRM blue + gold theme)
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --srm-blue: #17479e;
    --srm-blue-dark: #0f2f6b;
    --srm-blue-soft: #eaf0fb;
    --srm-gold: #c9a21a;
    --ink: #1c2540;
    --muted: #5b6785;
    --line: #dde4f2;
    --bg-soft: #f5f7fc;
}

html, body, [class*="css"], .stMarkdown, .stButton button, .stTextInput input {
    font-family: 'Inter', 'Segoe UI', sans-serif;
}

/* Layout */
.block-container {
    padding-top: 1.6rem !important;
    padding-bottom: 3rem !important;
    max-width: 1200px;
}
footer {visibility: hidden;}
#MainMenu {visibility: hidden;}
header[data-testid="stHeader"] {background: transparent;}

/* Hero banner */
.hero {
    background: linear-gradient(120deg, var(--srm-blue-dark) 0%, var(--srm-blue) 70%, #2a62c4 100%);
    color: #ffffff;
    border-radius: 18px;
    padding: 34px 38px;
    border-bottom: 6px solid var(--srm-gold);
    box-shadow: 0 10px 28px rgba(23, 71, 158, 0.22);
    margin-bottom: 22px;
}
.hero h1 {
    color: #ffffff !important;
    font-size: 2.3rem;
    font-weight: 800;
    margin: 0 0 6px 0;
    padding: 0;
    line-height: 1.2;
}
.hero p { color: #dbe6ff; font-size: 1.05rem; margin: 4px 0; }
.hero .badge {
    display: inline-block;
    background: rgba(255,255,255,0.14);
    border: 1px solid rgba(255,255,255,0.28);
    color: #ffffff;
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 600;
    margin: 12px 8px 0 0;
}

/* Compact page header */
.page-head {
    background: #ffffff;
    border: 1px solid var(--line);
    border-left: 6px solid var(--srm-gold);
    border-radius: 14px;
    padding: 18px 24px;
    margin-bottom: 18px;
    box-shadow: 0 3px 12px rgba(23, 71, 158, 0.06);
}
.page-head h2 { margin: 0; padding: 0; color: var(--srm-blue-dark); font-size: 1.6rem; font-weight: 800; }
.page-head p  { margin: 4px 0 0 0; color: var(--muted); font-size: 0.98rem; }

/* Cards */
.card {
    background: #ffffff;
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 20px 22px;
    box-shadow: 0 3px 12px rgba(23, 71, 158, 0.06);
    height: 100%;
    color: var(--ink);
}
.card h4 { margin: 0 0 6px 0; color: var(--srm-blue-dark); font-size: 1.05rem; font-weight: 700; }
.card p  { margin: 0; color: var(--muted); font-size: 0.93rem; line-height: 1.5; }
.card .icon { font-size: 1.8rem; margin-bottom: 8px; }

.step {
    display: flex; gap: 14px; align-items: flex-start;
    background: var(--bg-soft); border: 1px solid var(--line);
    border-radius: 12px; padding: 14px 16px; margin-bottom: 10px; color: var(--ink);
}
.step .num {
    flex: 0 0 34px; height: 34px; border-radius: 50%;
    background: var(--srm-blue); color: #ffffff; font-weight: 700;
    display: flex; align-items: center; justify-content: center;
}
.step b { color: var(--srm-blue-dark); }

.tip {
    background: #fff9e6; border: 1px solid #f0dc93; border-left: 5px solid var(--srm-gold);
    border-radius: 10px; padding: 12px 16px; color: #4a3d08; font-size: 0.93rem; margin: 8px 0 14px 0;
}

.mol-chip {
    display: inline-block; background: var(--srm-blue-soft); color: var(--srm-blue-dark);
    border-radius: 999px; padding: 4px 12px; font-weight: 600; font-size: 0.85rem; margin: 2px 6px 2px 0;
}

.dev-card {
    background: linear-gradient(120deg, #f5f8ff, #ffffff);
    border: 1px solid var(--line); border-radius: 14px; padding: 20px 24px; color: var(--ink);
}

/* Native Streamlit widgets */
[data-testid="stMetric"] {
    background: #ffffff; border: 1px solid var(--line); border-radius: 12px;
    padding: 14px 16px; box-shadow: 0 2px 8px rgba(23, 71, 158, 0.05);
}
[data-testid="stMetricLabel"] p { color: var(--muted) !important; font-weight: 600; }
[data-testid="stMetricValue"] { color: var(--srm-blue-dark) !important; font-weight: 800; }

.stButton > button, .stDownloadButton > button {
    border-radius: 10px; font-weight: 600; border: 1px solid var(--srm-blue);
    transition: all 0.15s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    transform: translateY(-1px); box-shadow: 0 4px 12px rgba(23, 71, 158, 0.18);
}
.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] {
    background: var(--srm-blue); color: #ffffff; border-color: var(--srm-blue);
}

.stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 2px solid var(--line); }
.stTabs [data-baseweb="tab"] {
    border-radius: 10px 10px 0 0; padding: 10px 18px; font-weight: 600; color: var(--muted);
}
.stTabs [aria-selected="true"] { color: var(--srm-blue) !important; background: var(--srm-blue-soft); }

/* Sidebar */
[data-testid="stSidebar"] { background: #f3f6fc; border-right: 1px solid var(--line); }
[data-testid="stSidebar"] div[role="radiogroup"] label {
    padding: 8px 12px; border-radius: 10px; margin-bottom: 2px; width: 100%;
}
[data-testid="stSidebar"] div[role="radiogroup"] label:hover { background: var(--srm-blue-soft); }
.side-mol {
    background: #ffffff; border: 1px solid var(--line); border-radius: 12px;
    padding: 10px 14px; font-size: 0.85rem; color: var(--ink);
}
.side-mol code { color: var(--srm-blue-dark); font-weight: 600; }

/* Result badges */
.pass-badge { color: #0b6b3a; font-weight: 700; }
.fail-badge { color: #b3261e; font-weight: 700; }

/* Mobile */
@media (max-width: 768px) {
    .hero { padding: 22px 20px; }
    .hero h1 { font-size: 1.6rem; }
    .block-container { padding-left: 1rem !important; padding-right: 1rem !important; }
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# SHARED MOLECULE STATE
# ============================================================

EXAMPLE_MOLECULES = {
    "Ethanol": "CCO",
    "Benzene": "c1ccccc1",
    "Phenol": "Oc1ccccc1",
    "Acetic Acid": "CC(=O)O",
    "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
}

PAGES = [
    "🏠 Home",
    "📖 Theory",
    "🧪 Molecule Lab",
    "⚖️ Compare Molecules",
    "📝 Assessment",
    "📄 Project Report",
]

st.session_state.setdefault("active_smiles", "CCO")
st.session_state.setdefault("page", PAGES[0])

# Re-assigning keeps these values alive even when the widget that owns them
# is not drawn on the current page (otherwise Streamlit deletes them).
st.session_state["active_smiles"] = st.session_state["active_smiles"]


def set_smiles(smiles):
    """Callback: put a SMILES string into the shared input box."""
    st.session_state["active_smiles"] = smiles


def go_to(page_name):
    """Callback: switch page from a button."""
    st.session_state["page"] = page_name


def get_molecule(smiles):
    """Convert SMILES into an RDKit molecule."""

    if not smiles or not smiles.strip():
        return None

    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        return mol

    except Exception:
        return None


# ============================================================
# CALCULATE SIGMA AND PI BONDS
# ============================================================

def calculate_bond_information(mol):
    """
    Calculate:
    - Heavy-atom bonds
    - Sigma (σ) bonds, including bonds to implicit hydrogens
    - Pi (π) bonds
    - Total bond order count (σ + π)
    """

    # Heavy-atom graph bonds (RDKit default molecule representation)
    heavy_atom_bonds = mol.GetNumBonds()

    # Count π bonds using a Kekulé representation for aromatic systems
    kekule_mol = Chem.Mol(mol)

    try:
        Chem.Kekulize(kekule_mol, clearAromaticFlags=True)
    except Exception:
        # If Kekulization is not possible, continue with the available form
        pass

    pi_bonds = 0

    for bond in kekule_mol.GetBonds():
        bond_type = bond.GetBondType()

        if bond_type == Chem.BondType.DOUBLE:
            pi_bonds += 1
        elif bond_type == Chem.BondType.TRIPLE:
            pi_bonds += 2
        elif bond_type == Chem.BondType.QUADRUPLE:
            pi_bonds += 3
        elif bond.GetIsAromatic():
            # Fallback for aromatic bonds if Kekulization did not clear them
            pi_bonds += 0.5

    # Add implicit hydrogens. Every atom-to-atom connection contains one σ bond.
    mol_with_h = Chem.AddHs(Chem.Mol(mol))
    sigma_bonds = mol_with_h.GetNumBonds()

    total_bonds = sigma_bonds + pi_bonds

    return {
        "Heavy Atom Bonds": heavy_atom_bonds,
        "Sigma Bonds": sigma_bonds,
        "Pi Bonds": pi_bonds,
        "Total Bonds": total_bonds,
        "Total Atoms Including H": mol_with_h.GetNumAtoms(),
    }


# ============================================================
# CALCULATE COMPLETE MOLECULAR PROPERTIES
# ============================================================

def calculate_properties(mol):

    # Calculate bond information
    bond_info = calculate_bond_information(mol)

    molecular_formula = rdMolDescriptors.CalcMolFormula(mol)

    properties = {

        "Molecular Formula": molecular_formula,

        "Molecular Weight": round(
            Descriptors.MolWt(mol),
            3
        ),

        "Exact Molecular Weight": round(
            Descriptors.ExactMolWt(mol),
            4
        ),

        "LogP": round(
            Crippen.MolLogP(mol),
            3
        ),

        "TPSA": round(
            rdMolDescriptors.CalcTPSA(mol),
            3
        ),

        "Hydrogen Bond Donors": Lipinski.NumHDonors(mol),

        "Hydrogen Bond Acceptors": Lipinski.NumHAcceptors(mol),

        "Rotatable Bonds": Lipinski.NumRotatableBonds(mol),

        "Ring Count": Lipinski.RingCount(mol),

        "Aromatic Rings": rdMolDescriptors.CalcNumAromaticRings(mol),

        "Aliphatic Rings": rdMolDescriptors.CalcNumAliphaticRings(mol),

        "Number of Atoms": mol.GetNumAtoms(),

        "Heavy Atoms": mol.GetNumHeavyAtoms(),

        # Bond information
        "Heavy Atom Bonds": bond_info["Heavy Atom Bonds"],

        "Sigma (σ) Bonds": bond_info["Sigma Bonds"],

        "Pi (π) Bonds": bond_info["Pi Bonds"],

        "Total Bonds (σ + π)": bond_info["Total Bonds"],

        "Fraction Csp3": round(
            rdMolDescriptors.CalcFractionCSP3(mol),
            3
        ),

        "Molar Refractivity": round(
            Crippen.MolMR(mol),
            3
        )
    }

    return properties

# ============================================================
# GENERATE 2D IMAGE
# ============================================================

def generate_2d_image(mol):

    image = Draw.MolToImage(
        mol,
        size=(700, 500)
    )

    return image


# ============================================================
# GENERATE 3D MOLECULE
# ============================================================

def generate_3d_molecule(mol):

    try:

        mol3d = Chem.AddHs(mol)

        params = AllChem.ETKDGv3()

        params.randomSeed = 42

        status = AllChem.EmbedMolecule(
            mol3d,
            params
        )

        if status != 0:
            return None

        try:

            AllChem.MMFFOptimizeMolecule(
                mol3d
            )

        except Exception:

            try:

                AllChem.UFFOptimizeMolecule(
                    mol3d
                )

            except Exception:
                pass

        return mol3d

    except Exception:

        return None


# ============================================================
# DISPLAY 3D STRUCTURE
# ============================================================

def display_3d_structure(mol3d, style, show_surface=False, spin=False, height=520):
    """Render an interactive 3D model (py3Dmol) inside the page."""

    mol_block = Chem.MolToMolBlock(mol3d)

    def build_html(width):
        view = py3Dmol.view(width=width, height=height)
        view.addModel(mol_block, "mol")

        if style == "Ball and Stick":
            view.setStyle({"stick": {"radius": 0.15}, "sphere": {"scale": 0.30}})
        elif style == "Stick":
            view.setStyle({"stick": {"radius": 0.20}})
        elif style == "Space Filling":
            view.setStyle({"sphere": {"scale": 1.0}})
        elif style == "Wireframe":
            view.setStyle({"line": {"linewidth": 2}})
        else:
            view.setStyle({"stick": {}, "sphere": {"scale": 0.20}})

        if show_surface:
            view.addSurface(py3Dmol.VDW, {"opacity": 0.7})

        if spin:
            view.spin(True)

        view.setBackgroundColor("white")
        view.zoomTo()
        return view._make_html()

    # Responsive width where the installed py3Dmol supports it; fixed otherwise.
    try:
        html = build_html("100%")
    except Exception:
        html = build_html(700)

    components.html(html, height=height + 20)


# ============================================================
# CREATE MOLECULAR PDF REPORT
# ============================================================

def generate_molecule_pdf(
    input_smiles,
    mol,
    properties
):

    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=45,
        rightMargin=45,
        topMargin=45,
        bottomMargin=45
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "TitleStyle",
        parent=styles["Title"],
        fontSize=20,
        alignment=TA_CENTER,
        textColor=HexColor("#17365D")
    )

    heading_style = ParagraphStyle(
        "HeadingStyle",
        parent=styles["Heading1"],
        fontSize=14,
        textColor=HexColor("#17365D")
    )

    body_style = ParagraphStyle(
        "BodyStyle",
        parent=styles["BodyText"],
        fontSize=10,
        leading=14,
        alignment=TA_JUSTIFY
    )

    story = []

    story.append(
        Paragraph(
            "MOLECULAR PROPERTIES REPORT",
            title_style
        )
    )

    story.append(
        Spacer(1, 0.3 * inch)
    )

    # --------------------------------------------------------
    # BASIC INFORMATION
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "1. Molecular Information",
            heading_style
        )
    )

    canonical_smiles = Chem.MolToSmiles(mol)

    basic_data = [

        ["Property", "Value"],

        ["Input SMILES", input_smiles],

        ["Canonical SMILES", canonical_smiles],

        [
            "Molecular Formula",
            properties["Molecular Formula"]
        ]
    ]

    basic_table = Table(
        basic_data,
        colWidths=[2 * inch, 4 * inch]
    )

    basic_table.setStyle(
        TableStyle([

            ("BACKGROUND", (0, 0), (-1, 0), HexColor("#17365D")),

            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),

            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),

            ("VALIGN", (0, 0), (-1, -1), "TOP"),

            ("PADDING", (0, 0), (-1, -1), 7)

        ])
    )

    story.append(basic_table)

    story.append(
        Spacer(1, 0.3 * inch)
    )

    # --------------------------------------------------------
    # PROPERTY TABLE
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "2. Molecular Properties",
            heading_style
        )
    )

    property_data = [
        ["Property", "Value"]
    ]

    for key, value in properties.items():

        property_data.append(
            [key, str(value)]
        )

    property_table = Table(
        property_data,
        colWidths=[3 * inch, 3 * inch],
        repeatRows=1
    )

    property_table.setStyle(
        TableStyle([

            ("BACKGROUND", (0, 0), (-1, 0), HexColor("#2F5597")),

            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),

            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),

            ("VALIGN", (0, 0), (-1, -1), "TOP"),

            ("PADDING", (0, 0), (-1, -1), 6)

        ])
    )

    story.append(property_table)

    story.append(
        Spacer(1, 0.3 * inch)
    )

    story.append(
        Paragraph(
            "3. Interpretation",
            heading_style
        )
    )

    interpretation = """
    This report was generated automatically using the
    Cheminformatics Virtual Laboratory. The molecular properties
    were calculated from the chemical structure represented by
    the submitted SMILES notation using the RDKit cheminformatics
    toolkit.
    """

    story.append(
        Paragraph(
            interpretation,
            body_style
        )
    )

    # --------------------------------------------------------
    # BUILD PDF
    # --------------------------------------------------------

    doc.build(story)

    pdf_data = buffer.getvalue()

    buffer.close()

    return pdf_data


# ============================================================
# REPORT IMAGE HELPERS
# ============================================================

def save_2d_structure_png(mol):
    """Create a temporary PNG for inclusion in PDF reports."""
    img = Draw.MolToImage(mol, size=(700, 500))
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    tmp.close()
    img.save(tmp.name)
    return tmp.name


def save_3d_structure_png(mol):
    """
    Create a high-clarity ball-and-stick image for the PDF report.

    The RDKit 3D coordinates are projected onto the best viewing plane so
    elongated molecules fill the page instead of appearing as a tiny object.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    mol3d = generate_3d_molecule(mol)
    if mol3d is None:
        return None

    conf = mol3d.GetConformer()
    n_atoms = mol3d.GetNumAtoms()

    # Collect 3D coordinates.
    coords = np.array([
        [
            conf.GetAtomPosition(i).x,
            conf.GetAtomPosition(i).y,
            conf.GetAtomPosition(i).z,
        ]
        for i in range(n_atoms)
    ], dtype=float)

    # Principal-component projection chooses the clearest overall view.
    centered = coords - coords.mean(axis=0)
    if n_atoms >= 3 and np.linalg.matrix_rank(centered) >= 2:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        projected = centered @ vh[:2].T
    else:
        projected = centered[:, :2]

    # CPK-style colours and readable atom sizes.
    cpk = {
        "H": "#FFFFFF", "C": "#3F3F3F", "N": "#2F5BFF",
        "O": "#E53935", "F": "#65C466", "P": "#F39C12",
        "S": "#F4D03F", "Cl": "#27AE60", "Br": "#8E3B2F",
        "I": "#8E44AD",
    }
    sizes = {
        "H": 70, "C": 240, "N": 270, "O": 285, "F": 260,
        "P": 300, "S": 310, "Cl": 310, "Br": 330, "I": 350,
    }

    fig, ax = plt.subplots(figsize=(8.8, 6.4), facecolor="white")
    ax.set_facecolor("white")

    # Draw bonds first. Multiple bonds are shown with parallel lines.
    for bond in mol3d.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        x1, y1 = projected[i]
        x2, y2 = projected[j]

        dx, dy = x2 - x1, y2 - y1
        length = float(np.hypot(dx, dy))
        if length == 0:
            continue

        # Perpendicular direction for double/triple bond separation.
        px, py = -dy / length, dx / length
        btype = bond.GetBondType()
        if btype == Chem.BondType.DOUBLE:
            offsets = (-0.035, 0.035)
        elif btype == Chem.BondType.TRIPLE:
            offsets = (-0.055, 0.0, 0.055)
        else:
            offsets = (0.0,)

        for off in offsets:
            ax.plot(
                [x1 + px * off, x2 + px * off],
                [y1 + py * off, y2 + py * off],
                color="#707070", linewidth=3.0,
                solid_capstyle="round", zorder=1
            )

    # Draw atoms over bonds.
    for i, atom in enumerate(mol3d.GetAtoms()):
        symbol = atom.GetSymbol()
        x, y = projected[i]
        ax.scatter(
            x, y,
            s=sizes.get(symbol, 250),
            c=cpk.get(symbol, "#A0A0A0"),
            edgecolors="#222222",
            linewidths=1.0,
            zorder=3
        )

        # Label important/non-carbon hetero atoms only; this keeps the model clean.
        if symbol not in ("H", "C"):
            ax.text(
                x, y, symbol,
                ha="center", va="center",
                fontsize=8, fontweight="bold",
                color="black", zorder=4
            )

    # Tight, equal-aspect limits make the molecule large and clear.
    xmin, ymin = projected.min(axis=0)
    xmax, ymax = projected.max(axis=0)
    width = max(xmax - xmin, 1.0)
    height = max(ymax - ymin, 1.0)
    pad = 0.22 * max(width, height)

    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    ax.set_title("3D Molecular Structure (Ball-and-Stick)", fontsize=18, fontweight="bold", pad=18)

    fig.tight_layout(pad=1.2)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    tmp.close()
    fig.savefig(
        tmp.name,
        dpi=320,
        bbox_inches="tight",
        facecolor="white",
        pad_inches=0.12
    )
    plt.close(fig)
    return tmp.name


# ============================================================
# PROPERTY INTERPRETATION HELPERS
# ============================================================

def get_property_interpretation(property_name, value, properties):
    """Return a student-friendly interpretation for each reported property."""

    if property_name == "Molecular Formula":
        return (
            "Shows the elemental composition of the molecule and is useful "
            "for understanding its basic chemical composition."
        )

    if property_name == "Molecular Weight":
        return (
            "Represents the average molecular mass. Molecular size and mass "
            "can influence diffusion, formulation and many physicochemical behaviours."
        )

    if property_name == "Exact Molecular Weight":
        return (
            "The monoisotopic mass calculated from exact isotopic masses; "
            "this is especially useful when comparing mass-spectrometry data."
        )

    if property_name == "LogP":
        if value < 0:
            return "A negative value indicates a preference toward the aqueous phase and relatively high hydrophilicity."
        elif value < 1:
            return "Indicates low to moderate lipophilicity, suggesting a reasonable preference for polar environments."
        elif value < 3:
            return "Indicates moderate lipophilicity, often reflecting a balance between hydrophilic and hydrophobic character."
        else:
            return "Indicates relatively high lipophilicity and a stronger preference for non-polar environments."

    if property_name == "TPSA":
        if value < 60:
            return "A relatively low polar surface area, indicating limited overall molecular polarity."
        elif value < 140:
            return "A moderate polar surface area, reflecting the contribution of heteroatoms to molecular polarity."
        else:
            return "A high polar surface area, indicating substantial polarity and strong hydrogen-bonding potential."

    if property_name == "Hydrogen Bond Donors":
        return (
            f"The molecule has {value} donor site(s) capable of donating hydrogen in hydrogen-bond interactions."
        )

    if property_name == "Hydrogen Bond Acceptors":
        return (
            f"The molecule has {value} acceptor site(s) capable of accepting hydrogen in hydrogen-bond interactions."
        )

    if property_name == "Rotatable Bonds":
        if value == 0:
            return "No freely rotatable bonds are detected, suggesting a relatively rigid molecular framework."
        return f"{value} rotatable bond(s) indicate conformational flexibility in the molecular structure."

    if property_name == "Ring Count":
        return f"The molecule contains {value} ring(s), which contribute to its overall shape and structural rigidity."

    if property_name == "Aromatic Rings":
        return f"{value} aromatic ring(s) contribute to aromaticity, planarity and delocalized π-electron character."

    if property_name == "Aliphatic Rings":
        return f"{value} non-aromatic ring(s) contribute to cyclic structure without aromatic π-electron delocalization."

    if property_name == "Number of Atoms":
        return "Counts atoms explicitly present in the RDKit molecular representation; implicit hydrogens are not included here."

    if property_name == "Heavy Atoms":
        return "Counts all non-hydrogen atoms and provides a useful measure of molecular structural size."

    if property_name == "Heavy Atom Bonds":
        return "Counts connections between non-hydrogen atoms in the molecular graph."

    if property_name == "Sigma (σ) Bonds":
        return "Every bonded atom pair contains one σ bond; this value includes bonds involving explicit or added hydrogens."

    if property_name == "Pi (π) Bonds":
        return "π bonds arise from multiple bonds and aromatic systems and are associated with electron delocalization."

    if property_name == "Total Bonds (σ + π)":
        return "Represents the total bond components obtained by adding the calculated σ and π bond counts."

    if property_name == "Fraction Csp3":
        if value < 0.25:
            return "A low sp³ fraction suggests a comparatively planar, unsaturated or aromatic structural character."
        elif value < 0.60:
            return "An intermediate sp³ fraction indicates a balance between saturated and unsaturated structural features."
        return "A high sp³ fraction indicates a more saturated, three-dimensional molecular framework."

    if property_name == "Molar Refractivity":
        return "Relates to molecular volume and electronic polarizability and reflects how the electron cloud responds to an electric field."

    return "This descriptor provides additional quantitative information about the molecular structure and physicochemical behaviour."


def build_key_interpretation_points(properties):
    """Create concise point-wise conclusions for the final report."""
    points = []

    points.append(
        f"<b>• Molecular identity:</b> The submitted structure has molecular formula "
        f"<b>{properties['Molecular Formula']}</b> and molecular weight "
        f"<b>{properties['Molecular Weight']}</b>."
    )

    logp = properties["LogP"]
    if logp < 0:
        lipophilic_text = "more hydrophilic than lipophilic"
    elif logp < 1:
        lipophilic_text = "low in lipophilicity"
    elif logp < 3:
        lipophilic_text = "moderately lipophilic"
    else:
        lipophilic_text = "strongly lipophilic"

    points.append(
        f"<b>• Lipophilicity:</b> LogP = <b>{logp}</b>, indicating that the molecule is "
        f"<b>{lipophilic_text}</b>. This descriptor describes the balance between "
        f"preference for non-polar and polar environments."
    )

    points.append(
        f"<b>• Molecular polarity:</b> TPSA = <b>{properties['TPSA']}</b> Å². "
        f"TPSA reflects the contribution of polar atoms and functional groups to the "
        f"overall surface polarity of the molecule."
    )

    points.append(
        f"<b>• Hydrogen bonding:</b> The structure contains "
        f"<b>{properties['Hydrogen Bond Donors']}</b> hydrogen-bond donor(s) and "
        f"<b>{properties['Hydrogen Bond Acceptors']}</b> acceptor(s), which helps "
        f"describe possible intermolecular interactions."
    )

    points.append(
        f"<b>• Flexibility:</b> {properties['Rotatable Bonds']} rotatable bond(s) "
        f"indicate the degree of conformational freedom available to the molecule."
    )

    points.append(
        f"<b>• Cyclic and aromatic character:</b> The molecule contains "
        f"{properties['Ring Count']} ring(s), including "
        f"{properties['Aromatic Rings']} aromatic ring(s). Aromatic rings are "
        f"associated with delocalized π-electron systems."
    )

    points.append(
        f"<b>• Bonding pattern:</b> The calculated structure contains "
        f"<b>{properties['Sigma (σ) Bonds']}</b> σ bonds and "
        f"<b>{properties['Pi (π) Bonds']}</b> π bonds. This helps distinguish the "
        f"framework-forming σ bonds from multiple-bond and delocalized π bonding."
    )

    points.append(
        f"<b>• Three-dimensional character:</b> Fraction Csp³ = "
        f"<b>{properties['Fraction Csp3']}</b>. This descriptor provides a simple "
        f"indication of the balance between saturated 3D carbon environments and "
        f"planar/unsaturated environments."
    )

    points.append(
        "<b>• Overall conclusion:</b> The calculated descriptors should be interpreted "
        "together with the 2D connectivity and generated 3D geometry to understand "
        "the molecule's composition, bonding, polarity, flexibility and structural character."
    )

    return points


# ============================================================
# GENERATE GENERAL FINAL PROJECT REPORT
# ============================================================

def generate_final_report(student_name, registration_number, input_smiles, mol, properties):
    """Generate a detailed personalized final report for the student's molecule."""

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=38,
        rightMargin=38,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ProjectTitle",
        parent=styles["Title"],
        fontSize=20,
        leading=25,
        alignment=TA_CENTER,
        textColor=HexColor("#17365D")
    )

    heading_style = ParagraphStyle(
        "ProjectHeading",
        parent=styles["Heading1"],
        fontSize=14,
        leading=18,
        textColor=HexColor("#17365D"),
        spaceBefore=8,
        spaceAfter=8
    )

    body_style = ParagraphStyle(
        "ProjectBody",
        parent=styles["BodyText"],
        fontSize=9.5,
        leading=13,
        alignment=TA_JUSTIFY
    )

    small_style = ParagraphStyle(
        "SmallTableText",
        parent=styles["BodyText"],
        fontSize=7.5,
        leading=9.5
    )

    header_style = ParagraphStyle(
        "HeaderTableText",
        parent=styles["BodyText"],
        fontSize=8,
        leading=9,
        textColor=colors.white,
        alignment=TA_CENTER
    )

    story = []

    story += [
        Spacer(1, 0.25 * inch),
        Paragraph("CHEMINFORMATICS VIRTUAL LABORATORY", title_style),
        Spacer(1, 0.12 * inch),
        Paragraph("FINAL STUDENT MOLECULAR ANALYSIS REPORT", heading_style),
        Spacer(1, 0.18 * inch)
    ]

    student_data = [
        ["Student Name", student_name or "Not provided"],
        ["Registration Number", registration_number or "Not provided"],
        ["Input SMILES", input_smiles],
        ["Canonical SMILES", Chem.MolToSmiles(mol)],
        ["Molecular Formula", properties["Molecular Formula"]]
    ]

    student_table = Table(student_data, colWidths=[2.0 * inch, 4.7 * inch])
    student_table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("BACKGROUND", (0, 0), (0, -1), HexColor("#EAF0F8")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("PADDING", (0, 0), (-1, -1), 6)
    ]))

    story += [student_table, Spacer(1, 0.22 * inch)]

    # --------------------------------------------------------
    # 1. STRUCTURES
    # --------------------------------------------------------
    story.append(Paragraph("1. Molecular Structures", heading_style))

    img2d = save_2d_structure_png(mol)
    img3d = save_3d_structure_png(mol)

    if img2d and img3d:
        structure_table = Table(
            [[
                Image(img2d, width=3.15 * inch, height=2.25 * inch),
                Image(img3d, width=3.15 * inch, height=2.25 * inch)
            ]],
            colWidths=[3.3 * inch, 3.3 * inch]
        )

        structure_table.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ALIGN", (0, 0), (-1, -1), "CENTER")
        ]))

        story += [
            structure_table,
            Spacer(1, 0.08 * inch),
            Paragraph(
                "<b>Figure 1.</b> Left: 2D molecular connectivity. "
                "Right: generated 3D ball-and-stick geometry showing the spatial arrangement of atoms.",
                body_style
            )
        ]

    elif img2d:
        story.append(Image(img2d, width=5.8 * inch, height=4.1 * inch))

    story += [Spacer(1, 0.18 * inch)]

    # --------------------------------------------------------
    # 2. COMPLETE PROPERTIES WITH INTERPRETATION COLUMN
    # --------------------------------------------------------
    story.append(Paragraph("2. Complete Molecular Properties and Interpretation", heading_style))

    prop_data = [[
        Paragraph("<b>Property</b>", header_style),
        Paragraph("<b>Value</b>", header_style),
        Paragraph("<b>Interpretation</b>", header_style)
    ]]

    for key, value in properties.items():
        interpretation = get_property_interpretation(key, value, properties)
        prop_data.append([
            Paragraph(str(key), small_style),
            Paragraph(str(value), small_style),
            Paragraph(interpretation, small_style)
        ])

    property_table = Table(
        prop_data,
        colWidths=[1.75 * inch, 1.0 * inch, 3.85 * inch],
        repeatRows=1
    )

    property_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#2F5597")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("PADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4)
    ]))

    story += [property_table, Spacer(1, 0.18 * inch)]

    # --------------------------------------------------------
    # 3. POINT-WISE INTERPRETATION
    # --------------------------------------------------------
    story.append(Paragraph("3. Point-wise Molecular Interpretation", heading_style))

    intro = (
        "The following points summarize the most important structural and "
        "physicochemical features calculated from the student's submitted SMILES."
    )
    story.append(Paragraph(intro, body_style))
    story.append(Spacer(1, 0.08 * inch))

    for point in build_key_interpretation_points(properties):
        story.append(Paragraph(point, body_style))
        story.append(Spacer(1, 0.06 * inch))

    # --------------------------------------------------------
    # 4. LEARNING NOTE
    # --------------------------------------------------------
    story += [
        Spacer(1, 0.12 * inch),
        Paragraph("4. Learning Note", heading_style),
        Paragraph(
            "The numerical descriptors in this report are computationally calculated "
            "from the submitted molecular structure. They are useful for molecular "
            "comparison and cheminformatics learning, but experimental measurements "
            "may differ depending on conditions and methodology.",
            body_style
        )
    ]


    # --------------------------------------------------------
    # 5. PYTHON LIBRARIES AND THEIR IMPORTANCE
    # --------------------------------------------------------
    story += [
        Spacer(1, 0.12 * inch),
        Paragraph("5. Python Libraries Used and Their Importance", heading_style),
        Paragraph(
            "The virtual laboratory uses the following libraries to process "
            "molecular structures, calculate descriptors, display results and "
            "create this PDF report.",
            body_style
        ),
        Spacer(1, 0.08 * inch)
    ]

    library_rows = [
        [
            Paragraph("<b>Library</b>", header_style),
            Paragraph("<b>Importance in this project</b>", header_style)
        ],
        [
            Paragraph("Streamlit", small_style),
            Paragraph("Builds the interactive web application, input fields, navigation, "
                      "visualization controls and download buttons.", small_style)
        ],
        [
            Paragraph("RDKit (Chem, Descriptors, Crippen, Lipinski, rdMolDescriptors, Draw, AllChem)",
                      small_style),
            Paragraph("Reads SMILES, validates molecular structures, generates 2D drawings "
                      "and 3D coordinates, and calculates molecular formula, molecular weight, "
                      "LogP, TPSA, hydrogen-bond counts, ring counts and other descriptors.",
                      small_style)
        ],
        [
            Paragraph("Pandas", small_style),
            Paragraph("Organizes molecular properties into tables and supports CSV export.",
                      small_style)
        ],
        [
            Paragraph("Matplotlib", small_style),
            Paragraph("Creates a clear 2D projection of the generated 3D molecular geometry "
                      "for inclusion in the PDF report.", small_style)
        ],
        [
            Paragraph("NumPy", small_style),
            Paragraph("Processes 3D atomic coordinates and uses numerical linear algebra "
                      "to select a clear projection of the molecule.", small_style)
        ],
        [
            Paragraph("py3Dmol", small_style),
            Paragraph("Displays interactive 3D molecular models in the web application.",
                      small_style)
        ],
        [
            Paragraph("ReportLab", small_style),
            Paragraph("Creates the downloadable PDF, including student details, structures, "
                      "property tables, interpretations, code examples and library notes.",
                      small_style)
        ],
        [
            Paragraph("Python standard libraries: io, os, tempfile", small_style),
            Paragraph("Manage in-memory PDF data and temporary image files, and clean up "
                      "temporary files after report generation.", small_style)
        ]
    ]

    library_table = Table(
        library_rows,
        colWidths=[2.05 * inch, 4.65 * inch],
        repeatRows=1
    )
    library_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#2F5597")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("PADDING", (0, 0), (-1, -1), 5)
    ]))
    story += [library_table, Spacer(1, 0.18 * inch)]

    # --------------------------------------------------------
    # 6. ILLUSTRATIVE PYTHON CODE
    # --------------------------------------------------------
    story.append(Paragraph("6. Python Code: SMILES to Molecular Properties", heading_style))
    story.append(Paragraph(
        "The following educational example shows the core workflow used by this "
        "laboratory. It is a compact example of the main calculation steps, not the "
        "complete Streamlit application source code.",
        body_style
    ))
    story.append(Spacer(1, 0.08 * inch))

    code_style = ParagraphStyle(
        "EducationalCode",
        parent=styles["Code"],
        fontName="Courier",
        fontSize=7,
        leading=8.5,
        leftIndent=5,
        rightIndent=5,
        spaceBefore=4,
        spaceAfter=4
    )

    educational_code = '''from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors, Draw
from rdkit.Chem import AllChem

# 1. Convert a SMILES string into a molecular object
smiles = "CCO"  # Example: ethanol
mol = Chem.MolFromSmiles(smiles)

if mol is None:
    raise ValueError("Invalid SMILES notation")

# 2. Calculate selected molecular descriptors
formula = rdMolDescriptors.CalcMolFormula(mol)
molecular_weight = Descriptors.MolWt(mol)
logp = Crippen.MolLogP(mol)
tpsa = rdMolDescriptors.CalcTPSA(mol)
h_bond_donors = Lipinski.NumHDonors(mol)
h_bond_acceptors = Lipinski.NumHAcceptors(mol)

print("Formula:", formula)
print("Molecular weight:", round(molecular_weight, 3), "g/mol")
print("LogP:", round(logp, 3))
print("TPSA:", round(tpsa, 3), "A^2")
print("H-bond donors:", h_bond_donors)
print("H-bond acceptors:", h_bond_acceptors)

# 3. Save a 2D molecular structure image
Draw.MolToFile(mol, "molecule_2d.png", size=(700, 500))

# 4. Generate a 3D conformer (coordinates)
mol_3d = Chem.AddHs(mol)
status = AllChem.EmbedMolecule(mol_3d, randomSeed=42)
if status == 0:
    AllChem.MMFFOptimizeMolecule(mol_3d)
    print("3D coordinates generated successfully")'''

    story.append(Preformatted(educational_code, code_style, maxLineLength=92))
    story.append(Spacer(1, 0.12 * inch))
    story.append(Paragraph("How to read the code", heading_style))
    code_notes = [
        "<b>SMILES input:</b> A text representation of the molecular connectivity.",
        "<b>Validation:</b> <font name='Courier'>Chem.MolFromSmiles()</font> attempts to parse the input; "
        "an invalid structure returns <font name='Courier'>None</font>.",
        "<b>Descriptors:</b> RDKit functions calculate numerical properties from the parsed molecular graph.",
        "<b>2D drawing:</b> <font name='Courier'>Draw.MolToFile()</font> writes a molecular diagram to an image file.",
        "<b>3D geometry:</b> Explicit hydrogens are added, a conformer is generated, and MMFF optimization "
        "is attempted to refine its geometry."
    ]
    for note in code_notes:
        story.append(Paragraph("• " + note, body_style))
        story.append(Spacer(1, 0.04 * inch))

    # --------------------------------------------------------
    # 7. REPRODUCIBILITY NOTE
    # --------------------------------------------------------
    story += [
        Spacer(1, 0.10 * inch),
        Paragraph("7. Reproducibility and Limitations", heading_style),
        Paragraph(
            "This code is an educational illustration of the principal workflow. "
            "The web application contains additional functions for student inputs, "
            "property interpretation, 3D display and PDF formatting. Calculated "
            "descriptors are computational estimates and should be interpreted with "
            "their definitions and, where appropriate, compared with experimental data.",
            body_style
        )
    ]


    doc.build(story)

    pdf_data = buffer.getvalue()
    buffer.close()

    for path in [img2d, img3d]:
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except Exception:
                pass

    return pdf_data



# ============================================================
# GENERATE ONE PDF FOR MULTIPLE MOLECULES
# ============================================================

def generate_multi_final_report(student_name, registration_number, molecule_records):
    """Create one PDF containing results for every valid SMILES supplied."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4, leftMargin=38, rightMargin=38,
        topMargin=40, bottomMargin=40
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "MultiReportTitle", parent=styles["Title"], fontSize=19,
        leading=23, alignment=TA_CENTER, textColor=HexColor("#17365D")
    )
    heading_style = ParagraphStyle(
        "MultiReportHeading", parent=styles["Heading1"], fontSize=13,
        leading=16, textColor=HexColor("#17365D"), spaceBefore=7, spaceAfter=7
    )
    body_style = ParagraphStyle(
        "MultiReportBody", parent=styles["BodyText"], fontSize=9,
        leading=12, alignment=TA_JUSTIFY
    )
    small_style = ParagraphStyle(
        "MultiReportSmall", parent=styles["BodyText"], fontSize=7.2, leading=8.8
    )
    header_style = ParagraphStyle(
        "MultiReportHeader", parent=styles["BodyText"], fontSize=7.5,
        leading=9, textColor=colors.white, alignment=TA_CENTER
    )
    story = [
        Spacer(1, 0.15 * inch),
        Paragraph("CHEMINFORMATICS VIRTUAL LABORATORY", title_style),
        Spacer(1, 0.08 * inch),
        Paragraph("MULTI-MOLECULE STUDENT ANALYSIS REPORT", heading_style),
        Paragraph(
            f"Student: {student_name or 'Not provided'} &nbsp;&nbsp;&nbsp; "
            f"Registration Number: {registration_number or 'Not provided'}",
            body_style
        ),
        Paragraph(f"Number of molecules analyzed: {len(molecule_records)}", body_style),
        Spacer(1, 0.15 * inch)
    ]

    temp_images = []
    for index, record in enumerate(molecule_records, start=1):
        input_smiles, mol, properties = record
        if index > 1:
            story.append(PageBreak())

        story.append(Paragraph(f"Molecule {index}: Molecular Information", heading_style))
        basic_data = [
            ["Input SMILES", input_smiles],
            ["Canonical SMILES", Chem.MolToSmiles(mol)],
            ["Molecular Formula", properties["Molecular Formula"]],
            ["Molecular Weight", str(properties["Molecular Weight"]) + " g/mol"]
        ]
        basic_table = Table(basic_data, colWidths=[1.8 * inch, 4.9 * inch])
        basic_table.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
            ("BACKGROUND", (0, 0), (0, -1), HexColor("#EAF0F8")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("PADDING", (0, 0), (-1, -1), 5)
        ]))
        story += [basic_table, Spacer(1, 0.12 * inch)]

        story.append(Paragraph("Molecular Structures", heading_style))
        img2d = save_2d_structure_png(mol)
        img3d = save_3d_structure_png(mol)
        if img2d:
            temp_images.append(img2d)
        if img3d:
            temp_images.append(img3d)

        if img2d and img3d:
            structure_table = Table([[
                Image(img2d, width=3.05 * inch, height=2.0 * inch),
                Image(img3d, width=3.05 * inch, height=2.0 * inch)
            ]], colWidths=[3.25 * inch, 3.25 * inch])
            structure_table.setStyle(TableStyle([
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (0, 0), (-1, -1), "CENTER")
            ]))
            story += [structure_table, Spacer(1, 0.12 * inch)]
        elif img2d:
            story += [Image(img2d, width=4.8 * inch, height=3.2 * inch),
                      Spacer(1, 0.12 * inch)]

        story.append(Paragraph("Molecular Properties and Interpretation", heading_style))
        prop_data = [[
            Paragraph("<b>Property</b>", header_style),
            Paragraph("<b>Value</b>", header_style),
            Paragraph("<b>Interpretation</b>", header_style)
        ]]
        for key, value in properties.items():
            prop_data.append([
                Paragraph(str(key), small_style),
                Paragraph(str(value), small_style),
                Paragraph(get_property_interpretation(key, value, properties), small_style)
            ])
        prop_table = Table(
            prop_data, colWidths=[1.55 * inch, 0.9 * inch, 4.0 * inch], repeatRows=1
        )
        prop_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), HexColor("#2F5597")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("PADDING", (0, 0), (-1, -1), 3)
        ]))
        story += [prop_table, Spacer(1, 0.12 * inch)]

        story.append(Paragraph("Key Interpretation", heading_style))
        for point in build_key_interpretation_points(properties):
            story.append(Paragraph(point, body_style))
            story.append(Spacer(1, 0.04 * inch))

    story.append(PageBreak())
    story.append(Paragraph("Python Libraries Used and Their Importance", heading_style))
    library_data = [
        [Paragraph("<b>Library</b>", header_style),
         Paragraph("<b>Importance in this project</b>", header_style)],
        [Paragraph("Streamlit", small_style),
         Paragraph("Creates the interactive web app, input controls and PDF download interface.", small_style)],
        [Paragraph("RDKit", small_style),
         Paragraph("Parses SMILES, validates structures, creates molecular drawings and 3D conformers, and calculates molecular descriptors.", small_style)],
        [Paragraph("Pandas", small_style),
         Paragraph("Organizes molecular properties in tables and supports CSV export.", small_style)],
        [Paragraph("Matplotlib and NumPy", small_style),
         Paragraph("Process molecular coordinates and generate a clear 2D projection of 3D geometry for the report.", small_style)],
        [Paragraph("py3Dmol", small_style),
         Paragraph("Displays interactive 3D molecular structures in the web application.", small_style)],
        [Paragraph("ReportLab", small_style),
         Paragraph("Builds the downloadable PDF with tables, figures and explanatory text.", small_style)],
        [Paragraph("io, os, tempfile", small_style),
         Paragraph("Handle in-memory PDF data and temporary image files.", small_style)]
    ]
    lib_table = Table(library_data, colWidths=[1.8 * inch, 4.9 * inch], repeatRows=1)
    lib_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#2F5597")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("PADDING", (0, 0), (-1, -1), 5)
    ]))
    story += [lib_table, Spacer(1, 0.15 * inch)]

    story.append(Paragraph("Illustrative Python Code", heading_style))
    code_style = ParagraphStyle(
        "MultiReportCode", parent=styles["Code"], fontName="Courier",
        fontSize=7, leading=8.5, leftIndent=4, rightIndent=4
    )
    code_example = "\n".join([
        "from rdkit import Chem",
        "from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors",
        "",
        'smiles_list = ["CCO", "c1ccccc1"]  # Add more SMILES strings here',
        "",
        "for smiles in smiles_list:",
        "    mol = Chem.MolFromSmiles(smiles)",
        "    if mol is None:",
        '        print("Invalid SMILES:", smiles)',
        "        continue",
        "",
        '    print("SMILES:", smiles)',
        '    print("Formula:", rdMolDescriptors.CalcMolFormula(mol))',
        '    print("Molecular weight:", Descriptors.MolWt(mol))',
        '    print("LogP:", Crippen.MolLogP(mol))'
    ])
    story.append(Preformatted(code_example, code_style, maxLineLength=92))
    story.append(Spacer(1, 0.08 * inch))
    story.append(Paragraph(
        "This compact code example demonstrates processing multiple SMILES in one run. "
        "The complete application includes additional code for visualization, descriptor "
        "interpretation, student information and PDF generation. Calculated descriptors "
        "are computational results and should be interpreted appropriately.",
        body_style
    ))

    try:
        doc.build(story)
        return buffer.getvalue()
    finally:
        buffer.close()
        for image_path in temp_images:
            try:
                if os.path.exists(image_path):
                    os.remove(image_path)
            except Exception:
                pass




# ============================================================
# UI HELPERS
# ============================================================

import html as _html


def html_block(markup):
    """Render HTML safely (blank lines / indentation would break Markdown)."""
    cleaned = "\n".join(line.strip() for line in markup.strip().splitlines() if line.strip())
    st.markdown(cleaned, unsafe_allow_html=True)


def esc(value):
    return _html.escape(str(value))


def page_header(title, subtitle):
    html_block(f"""
    <div class="page-head">
    <h2>{title}</h2>
    <p>{subtitle}</p>
    </div>
    """)


def show_df(df):
    """Full-width table that works on old and new Streamlit versions."""
    try:
        st.dataframe(df, width="stretch", hide_index=True)
    except Exception:
        st.dataframe(df, use_container_width=True, hide_index=True)


def show_image(img):
    try:
        st.image(img, width="stretch")
    except Exception:
        st.image(img, use_container_width=True)


def wide_button(label, **kwargs):
    try:
        return st.button(label, width="stretch", **kwargs)
    except Exception:
        return st.button(label, use_container_width=True, **kwargs)


def open_in_lab(smiles):
    """Callback: load a molecule and jump to the Molecule Lab."""
    st.session_state["active_smiles"] = smiles
    st.session_state["page"] = "🧪 Molecule Lab"


@st.cache_data(show_spinner=False)
def cached_3d_block(smiles):
    """Generate 3D coordinates once per molecule and reuse them."""
    mol = get_molecule(smiles)
    if mol is None:
        return None
    mol3d = generate_3d_molecule(mol)
    if mol3d is None:
        return None
    return Chem.MolToMolBlock(mol3d)


def lipinski_rules(props):
    """Lipinski Rule of Five + Veber rules, as rows for display."""
    rows = [
        ("Molecular weight", props["Molecular Weight"], "≤ 500", props["Molecular Weight"] <= 500, "Lipinski"),
        ("LogP", props["LogP"], "≤ 5", props["LogP"] <= 5, "Lipinski"),
        ("H-bond donors", props["Hydrogen Bond Donors"], "≤ 5", props["Hydrogen Bond Donors"] <= 5, "Lipinski"),
        ("H-bond acceptors", props["Hydrogen Bond Acceptors"], "≤ 10", props["Hydrogen Bond Acceptors"] <= 10, "Lipinski"),
        ("TPSA (Å²)", props["TPSA"], "≤ 140", props["TPSA"] <= 140, "Veber"),
        ("Rotatable bonds", props["Rotatable Bonds"], "≤ 10", props["Rotatable Bonds"] <= 10, "Veber"),
    ]
    return rows


def lipinski_violations(props):
    return sum(1 for r in lipinski_rules(props) if r[4] == "Lipinski" and not r[3])


def molecule_input(prefix):
    """Shared SMILES input with one-click example molecules. Returns the SMILES text."""
    st.markdown("##### 1️⃣ Pick an example molecule or type your own SMILES")

    names = list(EXAMPLE_MOLECULES.keys())
    cols = st.columns(len(names))
    active = st.session_state["active_smiles"].strip()

    for col, name in zip(cols, names):
        smi = EXAMPLE_MOLECULES[name]
        with col:
            wide_button(
                name,
                key=f"{prefix}_ex_{name}",
                on_click=set_smiles,
                args=(smi,),
                type="primary" if active == smi else "secondary",
            )

    st.text_input(
        "SMILES notation",
        key="active_smiles",
        placeholder="Example: CC(=O)Oc1ccccc1C(=O)O",
        help="SMILES is a short text code that describes a molecule. "
             "Try CCO (ethanol) or c1ccccc1 (benzene).",
    )
    return st.session_state["active_smiles"]


def invalid_smiles_help(smiles):
    st.error("❌ That SMILES could not be read, so no structure was drawn.")
    st.markdown(
        """
**Quick checklist**
- Are all **parentheses** `( )` and **brackets** `[ ]` closed?
- Does every **ring number** appear twice (e.g. `c1ccccc1`)?
- Aromatic atoms are **lowercase** (`c`, `n`, `o`); aliphatic atoms are uppercase (`C`, `N`, `O`).
- Use only valid element symbols and no spaces inside the string.
"""
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    if os.path.exists(LOGO_PATH):
        show_image(LOGO_PATH)

    st.markdown("### 🧪 Virtual Lab")

    st.radio(
        "Navigate",
        PAGES,
        key="page",
        label_visibility="collapsed",
    )

    st.divider()

    with st.expander(
        "👩‍🎓 Student information",
        expanded=(st.session_state["page"] == "📄 Project Report"),
    ):
        st.text_input("Student name", key="student_name")
        st.text_input("Registration number", key="registration_number")
        st.caption("Used on your PDF report.")

    _side_smiles = st.session_state["active_smiles"].strip()
    _side_mol = get_molecule(_side_smiles)
    if _side_mol is not None:
        _side_formula = rdMolDescriptors.CalcMolFormula(_side_mol)
        html_block(f"""
        <div class="side-mol">
        <b>Current molecule</b><br>
        <code>{esc(_side_smiles)}</code><br>
        {esc(_side_formula)}
        </div>
        """)

page = st.session_state["page"]
student_name = st.session_state.get("student_name", "")
registration_number = st.session_state.get("registration_number", "")

# Keep per-page inputs alive while the user visits other pages.
for _key in ("report_smiles", "compare_text"):
    if _key in st.session_state:
        st.session_state[_key] = st.session_state[_key]


# ============================================================
# HOME PAGE
# ============================================================

def render_home():

    if os.path.exists(LOGO_PATH):
        logo_col, _ = st.columns([1, 2])
        with logo_col:
            show_image(LOGO_PATH)

    html_block("""
    <div class="hero">
    <h1>🧪 Cheminformatics Virtual Laboratory</h1>
    <p>Draw, visualise and analyse molecules from a simple text code — right in your browser.</p>
    <p><b>SRM Institute of Science and Technology, Tiruchirappalli</b></p>
    <span class="badge">School of Sciences</span>
    <span class="badge">Division of Chemistry</span>
    <span class="badge">Faculty of Engineering and Technology</span>
    </div>
    """)

    c1, c2 = st.columns([1, 1])
    with c1:
        wide_button(
            "🚀 Start the Laboratory",
            key="home_start",
            type="primary",
            on_click=go_to,
            args=("🧪 Molecule Lab",),
        )
    with c2:
        wide_button(
            "📖 Read the Theory first",
            key="home_theory",
            on_click=go_to,
            args=("📖 Theory",),
        )

    st.markdown("### What you can do")

    features = [
        ("⌨️", "Enter SMILES", "Type a molecule as text and get instant validation with helpful error hints."),
        ("🖼️", "See it in 2D", "A clean structural drawing with formula, σ/π bond counts and atom counts."),
        ("🌐", "Explore it in 3D", "Rotate, zoom and switch between ball-and-stick, stick, space-filling and wireframe."),
        ("📊", "Calculate descriptors", "MW, LogP, TPSA, H-bond donors/acceptors, rotatable bonds, rings and more — each explained."),
        ("💊", "Check drug-likeness", "Automatic Lipinski Rule-of-Five and Veber checks with a pass/fail table."),
        ("⚖️", "Compare molecules", "Analyse many molecules together with charts and a downloadable table."),
    ]
    for row_start in (0, 3):
        cols = st.columns(3)
        for col, (icon, title, text) in zip(cols, features[row_start:row_start + 3]):
            with col:
                html_block(f"""
                <div class="card" style="min-height:165px;">
                <div class="icon">{icon}</div>
                <h4>{title}</h4>
                <p>{text}</p>
                </div>
                """)
        st.write("")

    st.markdown("### How to use this lab")
    left, right = st.columns([1, 1])
    with left:
        html_block("""
        <div class="step"><div class="num">1</div><div><b>Learn</b><br>Skim the Theory page to understand SMILES and descriptors.</div></div>
        <div class="step"><div class="num">2</div><div><b>Explore</b><br>Open the Molecule Lab and try a molecule from the examples.</div></div>
        """)
    with right:
        html_block("""
        <div class="step"><div class="num">3</div><div><b>Compare &amp; test</b><br>Compare several molecules, then take the 12-question assessment.</div></div>
        <div class="step"><div class="num">4</div><div><b>Report</b><br>Enter your details and download a combined PDF report.</div></div>
        """)

    st.markdown("### Try an example molecule")
    st.caption("Click a molecule to open it directly in the Molecule Lab.")

    cols = st.columns(len(EXAMPLE_MOLECULES))
    for col, (name, smi) in zip(cols, EXAMPLE_MOLECULES.items()):
        with col:
            wide_button(name, key=f"home_ex_{name}", on_click=open_in_lab, args=(smi,))

    st.write("")
    html_block("""
    <div class="dev-card">
    <b style="color:#0f2f6b;font-size:1.05rem;">👨‍🏫 Developed by</b><br>
    <b>Dr. R. Subramanian</b><br>
    Assistant Professor, Division of Chemistry, School of Sciences<br>
    Faculty of Engineering and Technology<br>
    SRM Institute of Science and Technology, Tiruchirappalli<br><br>
    📧 For queries, contact: <b>rsmani84@gmail.com</b>
    </div>
    """)


# ============================================================
# THEORY PAGE
# ============================================================

def render_theory():

    page_header(
        "📖 Introduction to Cheminformatics",
        "Short, friendly notes — read them before you start the laboratory.",
    )

    t1, t2, t3, t4, t5 = st.tabs([
        "What is it?", "SMILES", "Descriptors", "Drug-likeness", "3D models"
    ])

    with t1:
        st.markdown("""
**Cheminformatics** uses computers to store, process, visualise and analyse chemical information.
It combines three areas:

- ⚗️ **Chemistry** – structures, reactions and properties
- 💻 **Computer science** – text codes, algorithms and databases
- 📊 **Data analysis** – finding patterns that link structure to behaviour

**Why does it matter?** Drug discovery, materials design and environmental chemistry all use
cheminformatics to screen thousands of molecules quickly before any experiment is done.
""")

    with t2:
        st.markdown("""
**SMILES** stands for *Simplified Molecular Input Line Entry System*. It describes a molecule as a line of text.
""")
        st.markdown("##### The basic rules")
        rules = pd.DataFrame({
            "Feature": ["Atoms", "Double / triple bond", "Branches", "Rings", "Aromatic atoms", "Charged / special atoms"],
            "How it is written": ["C, N, O, S …", "=  and  #", "( )", "Matching digits", "lowercase letters", "[ ] brackets"],
            "Example": ["CCO", "O=C=O,  C#N", "CC(=O)O", "c1ccccc1", "c1ccccc1", "[NH4+]"],
        })
        show_df(rules)

        st.markdown("##### Common molecules")
        show_df(pd.DataFrame({
            "Molecule": list(EXAMPLE_MOLECULES.keys()),
            "SMILES": list(EXAMPLE_MOLECULES.values()),
        }))
        st.caption("Tip: the same molecule can be written in several valid ways. The lab also shows the *canonical* (standard) SMILES.")

    with t3:
        st.markdown("Molecular descriptors are numbers that summarise a structure.")
        show_df(pd.DataFrame({
            "Descriptor": [
                "Molecular weight (MW)", "LogP", "TPSA",
                "H-bond donors (HBD)", "H-bond acceptors (HBA)",
                "Rotatable bonds", "Ring count", "Fraction Csp3",
            ],
            "What it tells you": [
                "Size / mass of the molecule",
                "Preference for oil (lipophilic) vs water (hydrophilic)",
                "Polarity from N and O atoms (Å²)",
                "N–H / O–H groups that can donate a hydrogen bond",
                "N / O atoms that can accept a hydrogen bond",
                "Flexibility of the structure",
                "Number of rings",
                "Share of saturated carbons (3D character)",
            ],
        }))

    with t4:
        st.markdown("""
**Lipinski's Rule of Five** is a guideline for orally active drugs. A molecule is more likely to be drug-like if it has **at most one violation** of:

- Molecular weight ≤ 500
- LogP ≤ 5
- H-bond donors ≤ 5
- H-bond acceptors ≤ 10

**Veber's rules** add: TPSA ≤ 140 Å² and rotatable bonds ≤ 10.

These are *rules of thumb*, not laws — many successful drugs break them. The Molecule Lab checks them automatically.
""")

    with t5:
        st.markdown("""
| Model | Shows | Best for |
|---|---|---|
| 🟢 **Ball and Stick** | Atoms as spheres, bonds as sticks | Understanding connectivity and bond angles |
| 🟡 **Stick** | Bonds only | Emphasising the bonding framework |
| 🔵 **Space Filling** | Atoms at their van der Waals size | Approximate molecular volume and shape |
| ⚫ **Wireframe** | Thin lines | Simple, uncluttered view of large molecules |

**Atom colours:** carbon = grey, hydrogen = white, oxygen = red, nitrogen = blue.
""")

    st.write("")
    wide_button(
        "🧪 Practise in the Molecule Lab",
        key="theory_to_lab",
        type="primary",
        on_click=go_to,
        args=("🧪 Molecule Lab",),
    )


# ============================================================
# MOLECULE LAB  (2D, 3D, properties, drug-likeness, downloads)
# ============================================================

def render_lab():

    page_header(
        "🧪 Molecule Lab",
        "Enter a molecule once — see its 2D and 3D structure, properties and drug-likeness.",
    )

    smiles = molecule_input("lab")
    mol = get_molecule(smiles)

    if not smiles.strip():
        st.info("👆 Choose an example above or type a SMILES string to begin.")
        return

    if mol is None:
        invalid_smiles_help(smiles)
        return

    props = calculate_properties(mol)
    canonical = Chem.MolToSmiles(mol)
    bond_info = calculate_bond_information(mol)

    # Remember for the report page.
    st.session_state["latest_smiles"] = smiles
    st.session_state["latest_properties"] = props

    st.success("✅ Valid molecular structure")
    formula_txt = esc(props["Molecular Formula"])
    mw_txt = props["Molecular Weight"]
    html_block(f"""
    <span class="mol-chip">Formula: {formula_txt}</span>
    <span class="mol-chip">MW: {mw_txt} g/mol</span>
    <span class="mol-chip">Canonical: {esc(canonical)}</span>
    """)

    tab_2d, tab_3d, tab_props, tab_drug, tab_dl = st.tabs([
        "🖼️ 2D Structure",
        "🌐 3D Model",
        "📊 Properties",
        "💊 Drug-likeness",
        "⬇️ Downloads",
    ])

    # ---------------------------------------------------------- 2D
    with tab_2d:
        c1, c2 = st.columns([1.3, 1])
        with c1:
            show_image(generate_2d_image(mol))
        with c2:
            st.markdown("##### Molecular information")
            info = pd.DataFrame({
                "Item": [
                    "Input SMILES", "Canonical SMILES", "Formula",
                    "Heavy atoms", "Total atoms (with H)",
                    "Heavy-atom bonds", "Sigma (σ) bonds", "Pi (π) bonds",
                    "Total bonds (σ + π)",
                ],
                "Value": [
                    smiles.strip(), canonical, props["Molecular Formula"],
                    props["Heavy Atoms"], bond_info["Total Atoms Including H"],
                    bond_info["Heavy Atom Bonds"], bond_info["Sigma Bonds"],
                    bond_info["Pi Bonds"], bond_info["Total Bonds"],
                ],
            })
            info["Value"] = info["Value"].astype(str)
            show_df(info)

    # ---------------------------------------------------------- 3D
    with tab_3d:
        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            style = st.selectbox(
                "Display style",
                ["Ball and Stick", "Stick", "Space Filling", "Wireframe"],
                key="lab_3d_style",
            )
        with c2:
            surface = st.checkbox("van der Waals surface", key="lab_3d_surface")
        with c3:
            spin = st.checkbox("Auto-rotate", key="lab_3d_spin")

        show_3d = True
        if mol.GetNumHeavyAtoms() > 60:
            show_3d = st.checkbox(
                "This is a large molecule — tick to generate the 3D model (may take a few seconds)",
                key="lab_3d_large",
            )

        if show_3d:
            with st.spinner("Building 3D structure…"):
                block = cached_3d_block(smiles.strip())

            if block is None:
                st.error("Unable to generate a 3D structure for this molecule.")
            else:
                mol3d = Chem.MolFromMolBlock(block, removeHs=False)
                display_3d_structure(mol3d, style, surface, spin)
                st.caption(
                    "🖱️ Drag to rotate • Scroll to zoom • Right-drag (or Ctrl + drag) to move.  "
                    "Colours: C grey, H white, O red, N blue."
                )

    # ---------------------------------------------------------- Properties
    with tab_props:
        c = st.columns(4)
        c[0].metric("Molecular weight", f"{props['Molecular Weight']} g/mol")
        c[1].metric("LogP", props["LogP"])
        c[2].metric("TPSA", f"{props['TPSA']} Å²")
        c[3].metric("Ring count", props["Ring Count"])

        c = st.columns(4)
        c[0].metric("H-bond donors", props["Hydrogen Bond Donors"])
        c[1].metric("H-bond acceptors", props["Hydrogen Bond Acceptors"])
        c[2].metric("Rotatable bonds", props["Rotatable Bonds"])
        c[3].metric("Fraction Csp3", props["Fraction Csp3"])

        st.markdown("##### Full descriptor table")
        rows = [
            {
                "Property": key,
                "Value": str(value),
                "What it means": get_property_interpretation(key, value, props),
            }
            for key, value in props.items()
        ]
        show_df(pd.DataFrame(rows))

    # ---------------------------------------------------------- Drug-likeness
    with tab_drug:
        rule_rows = lipinski_rules(props)
        table = pd.DataFrame({
            "Rule set": [r[4] for r in rule_rows],
            "Property": [r[0] for r in rule_rows],
            "Value": [str(r[1]) for r in rule_rows],
            "Limit": [r[2] for r in rule_rows],
            "Result": ["✅ Pass" if r[3] else "❌ Fails" for r in rule_rows],
        })
        show_df(table)

        violations = lipinski_violations(props)
        if violations == 0:
            st.success("✅ No Lipinski violations — this molecule fits the Rule of Five.")
        elif violations == 1:
            st.info("🟡 One Lipinski violation — still generally considered drug-like.")
        else:
            st.warning(f"⚠️ {violations} Lipinski violations — oral bioavailability may be a concern.")

        st.caption("These are screening guidelines for teaching purposes, not proof of drug activity.")

    # ---------------------------------------------------------- Downloads
    with tab_dl:
        report_rows = [["Input SMILES", smiles.strip()], ["Canonical SMILES", canonical]]
        for key, value in props.items():
            report_rows.append([key, value])
        report_df = pd.DataFrame(report_rows, columns=["Property", "Value"])
        report_df["Value"] = report_df["Value"].astype(str)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### Spreadsheet (CSV)")
            st.download_button(
                "⬇️ Download properties (CSV)",
                data=report_df.to_csv(index=False).encode("utf-8"),
                file_name="Molecular_Properties_Report.csv",
                mime="text/csv",
                key="lab_csv_download",
            )
        with c2:
            st.markdown("##### Printable report (PDF)")
            if st.button("📄 Create PDF report", key="lab_pdf_btn"):
                with st.spinner("Creating PDF report…"):
                    st.session_state["lab_pdf"] = (
                        smiles.strip(),
                        generate_molecule_pdf(smiles, mol, props),
                    )

            cached = st.session_state.get("lab_pdf")
            if cached and cached[0] == smiles.strip():
                st.download_button(
                    "⬇️ Download PDF report",
                    data=cached[1],
                    file_name="Molecular_Properties_Report.pdf",
                    mime="application/pdf",
                    key="lab_pdf_download",
                )


# ============================================================
# COMPARE MOLECULES
# ============================================================

def render_compare():

    page_header(
        "⚖️ Compare Molecules",
        "Analyse several molecules side by side and see how structure changes properties.",
    )

    st.markdown(
        "Enter **one molecule per line** as `Name, SMILES` (or just a SMILES). Edit the example or paste your own."
    )

    default_data = """Ethanol,CCO
Benzene,c1ccccc1
Phenol,Oc1ccccc1
Acetic Acid,CC(=O)O
Aspirin,CC(=O)Oc1ccccc1C(=O)O
Caffeine,CN1C=NC2=C1C(=O)N(C(=O)N2C)C"""

    st.session_state.setdefault("compare_text", default_data)
    text = st.text_area("Molecules", key="compare_text", height=190)

    rows, invalid = [], []

    for n, line in enumerate(text.splitlines()[:50], start=1):
        line = line.strip()
        if not line:
            continue

        if "," in line:
            name, smi = line.split(",", 1)
            name, smi = name.strip(), smi.strip()
        else:
            name, smi = f"Molecule {n}", line

        mol = get_molecule(smi)
        if mol is None:
            invalid.append(line)
            continue

        p = calculate_properties(mol)
        rows.append({
            "Molecule": name or f"Molecule {n}",
            "SMILES": smi,
            "Formula": p["Molecular Formula"],
            "MW": p["Molecular Weight"],
            "LogP": p["LogP"],
            "TPSA": p["TPSA"],
            "HBD": p["Hydrogen Bond Donors"],
            "HBA": p["Hydrogen Bond Acceptors"],
            "Rotatable Bonds": p["Rotatable Bonds"],
            "Ring Count": p["Ring Count"],
            "Lipinski violations": lipinski_violations(p),
        })

    if invalid:
        st.warning("These lines could not be read and were skipped: " + "; ".join(invalid))

    if not rows:
        st.info("Add at least one valid molecule to see the comparison.")
        return

    df = pd.DataFrame(rows)

    st.success(f"✅ {len(df)} molecule(s) analysed")
    show_df(df)

    st.download_button(
        "⬇️ Download comparison table (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="structure_property_comparison.csv",
        mime="text/csv",
        key="compare_csv_download",
    )

    st.markdown("### 📈 Charts")

    numeric_cols = ["MW", "LogP", "TPSA", "HBD", "HBA", "Rotatable Bonds", "Ring Count"]
    c1, c2 = st.columns(2)

    with c1:
        prop = st.selectbox("Property to plot", numeric_cols, key="compare_prop")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(df["Molecule"], df[prop], color="#17479e")
        ax.set_ylabel(prop)
        ax.set_title(f"{prop} by molecule", fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with c2:
        st.markdown("**Molecular weight vs LogP**")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(df["MW"], df["LogP"], s=90, color="#c9a21a", edgecolor="#0f2f6b", zorder=3)
        for _, r in df.iterrows():
            ax.annotate(r["Molecule"], (r["MW"], r["LogP"]), xytext=(6, 5), textcoords="offset points", fontsize=8)
        ax.set_xlabel("Molecular weight (g/mol)")
        ax.set_ylabel("LogP")
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


# ============================================================
# ASSESSMENT
# ============================================================

QUESTIONS = [
    {"question": "What is the full form of SMILES?",
     "options": ["Simplified Molecular Input Line Entry System", "Standard Molecular Information Language Encoding System", "Simple Molecular Identification and Labeling System"],
     "answer": "Simplified Molecular Input Line Entry System",
     "explain": "SMILES = Simplified Molecular Input Line Entry System."},
    {"question": "What does a SMILES string represent?",
     "options": ["A molecular structure using text", "A spectroscopy technique", "A laboratory instrument"],
     "answer": "A molecular structure using text",
     "explain": "SMILES encodes atoms, bonds, branches and rings as plain text."},
    {"question": "Which SMILES represents ethanol?",
     "options": ["CCO", "c1ccccc1", "CC(=O)O"],
     "answer": "CCO",
     "explain": "CCO = two carbons and an –OH group."},
    {"question": "Which SMILES represents benzene?",
     "options": ["c1ccccc1", "CCO", "O=C=O"],
     "answer": "c1ccccc1",
     "explain": "Lowercase c atoms with matching ring digits describe an aromatic six-membered ring."},
    {"question": "In SMILES, parentheses are mainly used to indicate:",
     "options": ["Branches", "Molecular weight", "Atom colour"],
     "answer": "Branches",
     "explain": "Parentheses mark side branches, e.g. CC(=O)O."},
    {"question": "Which property is commonly associated with molecular lipophilicity?",
     "options": ["LogP", "TPSA", "HBD"],
     "answer": "LogP",
     "explain": "LogP measures partitioning between oil and water."},
    {"question": "What does TPSA describe?",
     "options": ["Topological Polar Surface Area", "Total Pi Surface Area", "Thermal Property Surface Analysis"],
     "answer": "Topological Polar Surface Area",
     "explain": "TPSA sums the surface contributed by polar atoms (mainly N and O)."},
    {"question": "What does HBA mean?",
     "options": ["Hydrogen Bond Acceptor", "Heavy Bond Atom", "Hydrogen Bond Analysis"],
     "answer": "Hydrogen Bond Acceptor",
     "explain": "HBA = hydrogen-bond acceptor sites."},
    {"question": "What does HBD mean?",
     "options": ["Hydrogen Bond Donor", "Heavy Bond Descriptor", "Hydrogen Bond Distance"],
     "answer": "Hydrogen Bond Donor",
     "explain": "HBD = hydrogen-bond donor sites such as O–H and N–H."},
    {"question": "A double bond contains how many pi (π) bonds?",
     "options": ["1", "2", "0"],
     "answer": "1",
     "explain": "A double bond = 1 σ + 1 π."},
    {"question": "A triple bond contains how many pi (π) bonds?",
     "options": ["2", "1", "3"],
     "answer": "2",
     "explain": "A triple bond = 1 σ + 2 π."},
    {"question": "Which model is useful for visualizing approximate molecular volume?",
     "options": ["Space Filling", "Wireframe only", "Text SMILES"],
     "answer": "Space Filling",
     "explain": "Space-filling spheres are drawn at van der Waals size, showing molecular volume."},
]


def reset_assessment():
    for i in range(len(QUESTIONS)):
        st.session_state.pop(f"assess_{i}", None)
    st.session_state.pop("assessment_options", None)
    st.session_state.pop("assessment_result", None)


def render_assessment():

    page_header(
        "📝 Cheminformatics Assessment",
        f"{len(QUESTIONS)} multiple-choice questions • about 5 minutes • you can retake it any time.",
    )

    if "assessment_options" not in st.session_state:
        st.session_state["assessment_options"] = [
            random.sample(q["options"], len(q["options"])) for q in QUESTIONS
        ]

    options = st.session_state["assessment_options"]
    result = st.session_state.get("assessment_result")

    if result is None:
        with st.form("assessment_form"):
            for i, q in enumerate(QUESTIONS):
                st.radio(
                    f"**Q{i + 1}.** {q['question']}",
                    options[i],
                    index=None,
                    key=f"assess_{i}",
                )
                st.write("")
            submitted = st.form_submit_button("✅ Submit assessment", type="primary")

        if submitted:
            answers = [st.session_state.get(f"assess_{i}") for i in range(len(QUESTIONS))]
            unanswered = [i + 1 for i, a in enumerate(answers) if a is None]
            if unanswered:
                st.warning("Please answer every question. Missing: Q" + ", Q".join(str(n) for n in unanswered))
            else:
                score = sum(1 for a, q in zip(answers, QUESTIONS) if a == q["answer"])
                st.session_state["assessment_result"] = {"answers": answers, "score": score}
                st.rerun()
        return

    # ---- Results view
    score = result["score"]
    total = len(QUESTIONS)
    pct = score / total * 100

    c1, c2 = st.columns(2)
    c1.metric("Your score", f"{score} / {total}")
    c2.metric("Percentage", f"{pct:.1f}%")
    st.progress(score / total)

    if pct >= 80:
        st.success("🎉 Excellent performance!")
        if not st.session_state.get("assessment_balloons_shown"):
            st.balloons()
            st.session_state["assessment_balloons_shown"] = True
    elif pct >= 50:
        st.info("👍 Good effort! Review the explanations below and try again.")
    else:
        st.warning("📚 Please review the Theory section and try again.")

    st.markdown("### Review your answers")
    for i, (q, given) in enumerate(zip(QUESTIONS, result["answers"])):
        correct = given == q["answer"]
        with st.expander(f"{'✅' if correct else '❌'}  Q{i + 1}. {q['question']}", expanded=not correct):
            st.write(f"**Your answer:** {given}")
            if not correct:
                st.write(f"**Correct answer:** {q['answer']}")
            st.caption(q["explain"])

    def _retake():
        reset_assessment()
        st.session_state.pop("assessment_balloons_shown", None)

    st.button("🔄 Retake assessment", on_click=_retake, key="assessment_retake", type="primary")


# ============================================================
# PROJECT REPORT
# ============================================================

def use_current_molecule():
    current = st.session_state.get("active_smiles", "").strip()
    existing = st.session_state.get("report_smiles", "").strip()
    if current and current not in existing.splitlines():
        st.session_state["report_smiles"] = (existing + "\n" + current).strip()


def render_report():

    page_header(
        "📄 Final Project Report",
        "Combine every molecule you studied into a single downloadable PDF.",
    )

    if not (student_name.strip() and registration_number.strip()):
        st.warning(
            "👈 Add your **name** and **registration number** in the sidebar "
            "(Student information) so they appear on the report."
        )
    else:
        st.success(f"Report for **{student_name}** ({registration_number})")

    st.session_state.setdefault(
        "report_smiles",
        st.session_state.get("latest_smiles", st.session_state["active_smiles"]),
    )

    st.markdown("##### Molecules to include (one SMILES per line)")
    st.text_area(
        "SMILES list",
        key="report_smiles",
        height=150,
        label_visibility="collapsed",
        help="Example: CCO on the first line, c1ccccc1 on the second line, and so on.",
    )
    st.button("➕ Add the molecule from the Molecule Lab", on_click=use_current_molecule, key="report_add_current")

    entries = [l.strip() for l in st.session_state["report_smiles"].splitlines() if l.strip()]
    records, invalid = [], []

    for smi in entries:
        m = get_molecule(smi)
        if m is None:
            invalid.append(smi)
            continue
        records.append((smi, m, calculate_properties(m)))

    if invalid:
        st.warning("These SMILES could not be read and will not be included: " + ", ".join(invalid))

    if not records:
        st.info("Enter at least one valid SMILES string to generate the report.")
        return

    st.success(f"{len(records)} valid molecule(s) ready for the report.")
    preview = pd.DataFrame({
        "No.": list(range(1, len(records) + 1)),
        "SMILES": [r[0] for r in records],
        "Formula": [r[2]["Molecular Formula"] for r in records],
        "MW (g/mol)": [r[2]["Molecular Weight"] for r in records],
    })
    show_df(preview)

    signature = (tuple(r[0] for r in records), student_name, registration_number)

    if st.button("📄 Generate PDF report", type="primary", key="report_generate"):
        with st.spinner(f"Generating a PDF for {len(records)} molecule(s)…"):
            st.session_state["report_pdf"] = (
                signature,
                generate_multi_final_report(student_name, registration_number, records),
            )

    cached = st.session_state.get("report_pdf")
    if cached and cached[0] == signature:
        st.success("✅ Your report is ready.")
        st.download_button(
            "⬇️ Download combined report (PDF)",
            data=cached[1],
            file_name="Cheminformatics_Multi_Molecule_Report.pdf",
            mime="application/pdf",
            key="report_download",
        )


# ============================================================
# ROUTER
# ============================================================

ROUTES = {
    "🏠 Home": render_home,
    "📖 Theory": render_theory,
    "🧪 Molecule Lab": render_lab,
    "⚖️ Compare Molecules": render_compare,
    "📝 Assessment": render_assessment,
    "📄 Project Report": render_report,
}

ROUTES.get(page, render_home)()


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🧪 Cheminformatics Virtual Laboratory  |  SRM Institute of Science and Technology, Tiruchirappalli  |  "
    "Built with Python, Streamlit and RDKit"
)
