"""Verified catalogue of knowledge-base documents.

Every entry below was produced by inspecting the actual file in ``data/``
(title page, gazette header, publisher credits, PDF metadata, extractability)
on 2026-11-04. Nothing here is guessed from the file name alone.

``source_url`` policy: where a specific official page was confirmed, it is
used and ``source_url_precision`` is ``"document_page"``. Otherwise the
official publisher portal is given and ``source_url_precision`` is
``"publisher_portal"`` — we never invent a deep link.

Classification vocabulary
-------------------------
document_type:
    primary_legislation        Act of Parliament (binding central law)
    model_legislation          Model law circulated to States/UTs; NOT binding
                               unless a State/UT enacts it
    government_guidelines      Advisory national guidelines (non-statutory)
    government_research        Methodology / research publication
    secondary_commentary       Third-party summary; not authoritative
    simulated_market_commentary  Project-authored synthetic notes (mock data)
source_type:
    official_government | secondary | simulated
"""

from __future__ import annotations

from typing import Any

# Bump when catalogue metadata changes so ingestion re-writes chunk metadata.
CATALOG_VERSION = "2026-11-04.1"

INDIA_CODE = "https://www.indiacode.nic.in/"
MOHUA = "https://mohua.gov.in/"

# Keyed by file name inside data/.
OFFICIAL_DOCUMENTS: dict[str, dict[str, Any]] = {
    "reraact.pdf": {
        "document_id": "rera_act_2016",
        "document_title": "The Real Estate (Regulation and Development) Act, 2016 (Act No. 16 of 2016)",
        "publisher": "Ministry of Law and Justice (Legislative Department), Gazette of India Extraordinary, Part II Section 1, No. 17",
        "source_url": INDIA_CODE,
        "source_url_precision": "publisher_portal",
        "document_type": "primary_legislation",
        "source_type": "official_government",
        "binding_status": "binding_central_law",
        "jurisdiction": "India (central Act). Rules, Regulatory Authorities and portals are State/UT-specific.",
        "effective_date": "2016-05-01",
        "version": "As enacted — Gazette dated 2016-03-26 (assent 2016-03-25). Not a consolidated/amended text.",
        "section_style": "legislation",
        "include_in_rag": True,
        "notes": [
            "Original gazette text: s.1(2) still reads 'except the State of Jammu and Kashmir', "
            "superseded by the J&K Reorganisation Act, 2019. Verify current extent on India Code.",
            "Sections were brought into force in phases (some from 2016-05-01, remainder 2017-05-01).",
            "Pages 38-39 carry little or no extractable text.",
        ],
    },
    "CP Act 2019_1732700731.pdf": {
        "document_id": "consumer_protection_act_2019",
        "document_title": "The Consumer Protection Act, 2019 (Act No. 35 of 2019)",
        "publisher": "Ministry of Law and Justice (Legislative Department), Gazette of India Extraordinary, Part II Section 1, No. 54",
        "source_url": "https://consumeraffairs.nic.in/",
        "source_url_precision": "publisher_portal",
        "document_type": "primary_legislation",
        "source_type": "official_government",
        "binding_status": "binding_central_law",
        "jurisdiction": "India (central Act)",
        "effective_date": "2020-07-20",
        "version": "As enacted — Gazette dated 2019-08-09. Not a consolidated/amended text.",
        "section_style": "legislation",
        "include_in_rag": True,
        "notes": [
            "Commencement was by notification; most provisions were notified from 2020-07-20. Verify on India Code.",
            "Gazette text s.1(2) still excludes J&K (pre-reorganisation wording).",
        ],
    },
    "registerationact.pdf": {
        "document_id": "registration_act_1908",
        "document_title": "The Registration Act, 1908 (Act No. 16 of 1908)",
        "publisher": "India Code (Legislative Department, Ministry of Law and Justice) — consolidated text",
        "source_url": INDIA_CODE,
        "source_url_precision": "publisher_portal",
        "document_type": "primary_legislation",
        "source_type": "official_government",
        "binding_status": "binding_central_law",
        "jurisdiction": "India (central Act) with State amendments reproduced in the text; registration procedure and fees are State-administered.",
        "effective_date": "1909-01-01",
        "version": "India Code consolidated text including State amendments (PDF generated 2024-06-19).",
        "section_style": "legislation",
        "include_in_rag": True,
        "notes": [
            "File is named 'registerationact.pdf' (sic) in data/, not 'registrationact.pdf'.",
            "Watermarked 'IndiaCode'; first pages are the Arrangement of Sections.",
        ],
    },
    "ind197502.pdf": {
        "document_id": "indian_easements_act_1882",
        "document_title": "The Indian Easements Act, 1882 (Act No. 5 of 1882)",
        "publisher": "India Code (Legislative Department, Ministry of Law and Justice) — consolidated text",
        "source_url": INDIA_CODE,
        "source_url_precision": "publisher_portal",
        "document_type": "primary_legislation",
        "source_type": "official_government",
        "binding_status": "binding_central_law",
        "jurisdiction": "India (central Act) — applies only in the territories listed in s.1; does not extend to every State.",
        "effective_date": "1882-07-01",
        "version": "India Code consolidated text (PDF generated 2020-02-24).",
        "section_style": "legislation",
        "include_in_rag": True,
        "notes": ["Verified as the Indian Easements Act, 1882 from the title page."],
    },
    "mohua.pdf": {
        "document_id": "model_tenancy_act_2021",
        "document_title": "The Model Tenancy Act, 2021",
        "publisher": "Ministry of Housing and Urban Affairs, Government of India",
        "source_url": MOHUA,
        "source_url_precision": "publisher_portal",
        "document_type": "model_legislation",
        "source_type": "official_government",
        "binding_status": "model_law_not_binding_unless_adopted_by_state",
        "jurisdiction": "Model law for adoption by States/UTs. Applies only where a State/UT has enacted it or a law based on it.",
        "effective_date": "2021-06-02",
        "version": "Approved by Union Cabinet 2021-06-02 (PDF created 2021-06-02).",
        "section_style": "legislation",
        "include_in_rag": True,
        "notes": [
            "Verified as the Model Tenancy Act, 2021 (body heading 'THE MODEL TENANCY ACT, 2021').",
            "Uses 'clauses' rather than 'sections' because it is a model draft.",
        ],
    },
    "model building bye laws.pdf": {
        "document_id": "model_building_bye_laws_2016",
        "document_title": "Model Building Bye-Laws, 2016",
        "publisher": "Town and Country Planning Organisation, Ministry of Urban Development (now MoHUA), Government of India",
        "source_url": MOHUA,
        "source_url_precision": "publisher_portal",
        "document_type": "model_legislation",
        "source_type": "official_government",
        "binding_status": "model_law_not_binding_unless_adopted_by_state",
        "jurisdiction": "Model bye-laws for States/ULBs/Development Authorities. Local building bye-laws and DCRs prevail.",
        "effective_date": "2016",
        "version": "Revised and published 2016 (PDF modified 2019-07-11).",
        "section_style": "numbered_headings",
        "include_in_rag": True,
        "notes": ["Some pages are drawings/blank with little extractable text."],
    },
    "Volume-I Main URDPFI Guidelines 2014a.pdf": {
        "document_id": "urdpfi_guidelines_2014_vol1",
        "document_title": "Urban and Regional Development Plans Formulation and Implementation (URDPFI) Guidelines, 2014 — Volume I",
        "publisher": "Ministry of Urban Development (now MoHUA), Government of India",
        "source_url": MOHUA,
        "source_url_precision": "publisher_portal",
        "document_type": "government_guidelines",
        "source_type": "official_government",
        "binding_status": "advisory_guidelines",
        "jurisdiction": "National advisory planning guidelines; statutory plans are made by State/local planning authorities.",
        "effective_date": "2014",
        "version": "2014 edition, Volume I (PDF created 2014-06-16).",
        "section_style": "numbered_headings",
        "include_in_rag": True,
        "notes": [
            "Embedded fonts have no Unicode mapping: ~203 of 209 pages extract as glyph garbage. "
            "Requires OCR; OCR text may merge some words.",
        ],
    },
    "WHITE-PAPER-NHB-RESIDEX-METHODOLOGY.pdf": {
        "document_id": "nhb_residex_methodology_2020",
        "document_title": "White Paper on NHB RESIDEX Methodology (Housing Price Indices)",
        "publisher": "National Housing Bank (approved by the NHB RESIDEX Technical Advisory Committee)",
        "source_url": "https://www.nhb.org.in/about-residex/",
        "source_url_precision": "publisher_portal",
        "document_type": "government_research",
        "source_type": "official_government",
        "binding_status": "methodology_reference",
        "jurisdiction": "India — methodology for national housing price indices; not a legal instrument.",
        "effective_date": "2020-07-15",
        "version": "TAC composition as on 2020-07-15 (PDF created 2020-08-05).",
        "section_style": "numbered_headings",
        "include_in_rag": True,
        "notes": ["NHB disclaimer: index is for information, not investment advice."],
    },
    "rera_act_2016_explainer.txt": {
        "document_id": "rera_explainer_ipleaders_2022",
        "document_title": "Real Estate (Regulation and Development) Act, 2016 — explainer article (iPleaders blog, 2022-05-21)",
        "publisher": "iPleaders blog (third-party)",
        "source_url": "",
        "source_url_precision": "not_recorded",
        "document_type": "secondary_commentary",
        "source_type": "secondary",
        "binding_status": "secondary_commentary_not_authoritative",
        "jurisdiction": "Commentary on central RERA Act",
        "effective_date": "2022-05-21",
        "version": "Blog article text as saved in data/.",
        "section_style": "none",
        "include_in_rag": True,
        "notes": [
            "Secondary commentary — never cite as the law; always prefer reraact.pdf.",
            "Overlaps with reraact.pdf and the simulated rera_act_summary.txt primer.",
            "Original URL was not recorded in the file and is not reproduced here.",
        ],
    },
    "Income_Tax_Act_1961.pdf": {
        "document_id": "income_tax_act_1961_fa2008",
        "document_title": "Income-tax Act, 1961 (as amended by Finance Act, 2008)",
        "publisher": "Commercial compilation (contains Taxmann editorial footnotes); not an official India Code text",
        "source_url": "https://www.incometaxindia.gov.in/",
        "source_url_precision": "publisher_portal",
        "document_type": "primary_legislation",
        "source_type": "official_government",
        "binding_status": "outdated_do_not_rely",
        "jurisdiction": "India (central Act)",
        "effective_date": "1962-04-01",
        "version": "As amended by Finance Act, 2008 — about 18 years out of date.",
        "section_style": "legislation",
        "include_in_rag": False,
        "exclusion_reason": (
            "OUTDATED: reflects amendments only up to Finance Act 2008, and the Income-tax Act, 1961 has been "
            "replaced by the Income-tax Act, 2025 with effect from 2026-04-01 (transitional provisions apply to "
            "earlier years). Excluded so the system never gives tax guidance from a stale text."
        ),
        "notes": ["1,065 pages, fully text-extractable."],
    },
}

# Project-authored synthetic notes that already lived in the package. They are
# SIMULATED and must never be presented as real market observations or law.
SIMULATED_CORPUS_DEFAULTS: dict[str, Any] = {
    "publisher": "Property Investment Advisor project (synthetic illustrative note)",
    "source_url": "",
    "source_url_precision": "not_applicable",
    "document_type": "simulated_market_commentary",
    "source_type": "simulated",
    "binding_status": "simulated_not_evidence",
    "effective_date": "not_applicable",
    "version": "project mock dataset",
    "section_style": "none",
    "include_in_rag": True,
}

SIMULATED_JURISDICTION = {
    "hinjewadi_bearish_analyst_note.txt": "Pune (Hinjewadi) — simulated",
    "hinjewadi_bullish_analyst_note.txt": "Pune (Hinjewadi) — simulated",
    "mumbai_monsoon_flood_risk.txt": "Mumbai — simulated",
    "metro_expansion_plans.txt": "Multiple Indian cities — simulated",
}

# Documents required by the knowledge-base plan that are NOT present in data/.
MISSING_DOCUMENTS: list[dict[str, str]] = [
    {
        "expected_file": "transferofproperty.pdf",
        "document_title": "The Transfer of Property Act, 1882 (Act No. 4 of 1882)",
        "publisher": "India Code (Legislative Department, Ministry of Law and Justice)",
        "official_source_url": "https://www.indiacode.nic.in/handle/123456789/2338",
        "impact": "Sale, mortgage, lease and transfer rules (e.g. ss.54, 58, 105) cannot be cited.",
    },
    {
        "expected_file": "(none supplied)",
        "document_title": "The Indian Stamp Act, 1899 (Act No. 2 of 1899)",
        "publisher": "India Code (Legislative Department, Ministry of Law and Justice)",
        "official_source_url": "https://www.indiacode.nic.in/handle/123456789/15510",
        "impact": "Stamp duty cannot be cited. Rates are State-specific in any case (State Stamp Acts/schedules).",
    },
    {
        "expected_file": "(none supplied)",
        "document_title": "Real Estate (Regulation and Development) (General) Rules, 2016 — central rules for Union Territories without legislature (notified 2016-10-31)",
        "publisher": "Ministry of Housing and Urban Poverty Alleviation (now MoHUA)",
        "official_source_url": MOHUA,
        "impact": "Central RERA rules are absent; they apply only to certain UTs. Each State notifies its own RERA rules — none are in the knowledge base.",
    },
    {
        "expected_file": "(none supplied)",
        "document_title": "Income-tax Act, 2025 (in force from 2026-04-01) and Income-tax Rules, 2026",
        "publisher": "Income Tax Department / CBDT",
        "official_source_url": "https://www.incometaxindia.gov.in/",
        "impact": "Current tax treatment (rental income, capital gains) cannot be assessed; the 1961 copy supplied is outdated.",
    },
]


def simulated_metadata(filename: str) -> dict[str, Any]:
    stem = filename.rsplit(".", 1)[0]
    return {
        **SIMULATED_CORPUS_DEFAULTS,
        "document_id": f"sim_{stem}",
        "document_title": f"[SIMULATED] {stem.replace('_', ' ').title()}",
        "jurisdiction": SIMULATED_JURISDICTION.get(filename, "India — simulated"),
        "notes": ["Synthetic project data; not an observation of the real market."],
    }


def catalogue_summary() -> list[dict[str, str]]:
    """Compact list used in agent prompts and reports."""
    rows = []
    for fname, meta in OFFICIAL_DOCUMENTS.items():
        rows.append(
            {
                "file": fname,
                "title": meta["document_title"],
                "document_type": meta["document_type"],
                "binding_status": meta["binding_status"],
                "in_knowledge_base": "yes" if meta["include_in_rag"] else "no (" + meta.get("exclusion_reason", "")[:60] + ")",
            }
        )
    return rows
