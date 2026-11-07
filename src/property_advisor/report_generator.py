"""Report Generation Tool.

Generates the final, human-approved investment report in three formats:
reports/<name>.json, reports/<name>.md, reports/<name>.pdf.

This is only ever called AFTER human approval — never from
inside the graph itself (report["status"] must be "approved").
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

PROJECT_TITLE = "Property Alpha AI — Indian Property Investment Advisory Report"

DECISION_COLORS = {
    "BUY": colors.HexColor("#1e7e34"),
    "HOLD": colors.HexColor("#b8860b"),
    "AVOID": colors.HexColor("#c82333"),
}


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug or "report"


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _fmt_inr(value: Any) -> str:
    try:
        return f"INR {float(value):,.0f}"
    except (TypeError, ValueError):
        return str(value)


# --------------------------------------------------------------------------
# JSON
# --------------------------------------------------------------------------


def _write_json(report: dict[str, Any], path: str, generated_at: str) -> None:
    payload = {**report, "generated_at": generated_at}
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)


# --------------------------------------------------------------------------
# Markdown
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# Due-diligence / provenance sections (two-agent refactor)
# --------------------------------------------------------------------------


def _research_lines(report: dict[str, Any]) -> list[str]:
    rf = report.get("research_findings", {}) or {}
    lines = [
        "## Due Diligence Findings (Property Research & Due Diligence Agent)",
        f"- Research status: {rf.get('status', 'N/A')}",
        "",
        rf.get("summary", "") or "_No research summary available._",
        "",
        "### Regulatory Requirements",
    ]
    reqs = rf.get("regulatory_requirements", [])
    if reqs:
        for r in reqs:
            lines.append(
                f"- {r['requirement']} — {r.get('relevance', '')} "
                f"[{', '.join(r.get('citation_ids', []))}; binding: {', '.join(r.get('binding_status', []))}]"
            )
    else:
        lines.append("- None identified with valid citations")
    lines += ["", "### Property Concerns"]
    lines += [
        f"- ({c['severity']}, basis: {c['basis']}) {c['concern']}"
        + (f" [{', '.join(c['citation_ids'])}]" if c.get("citation_ids") else "")
        for c in rf.get("concerns", [])
    ] or ["- None flagged"]
    lines += ["", "### Evidence Gaps (missing / outdated / State-specific)"]
    lines += [f"- {g['gap']} — {g.get('why_it_matters', '')}" for g in rf.get("evidence_gaps", [])] or ["- None"]
    lines += ["", "### Conflicting Evidence"]
    lines += [f"- {c}" for c in rf.get("conflicting_evidence", [])] or ["- None detected"]
    if rf.get("unsupported_statements"):
        lines += ["", "### Statements Dropped (no valid citation)"]
        lines += [f"- {u['statement']}" for u in rf["unsupported_statements"]]
    lines += ["", "### Due Diligence Checklist"]
    lines += [f"- {c}" for c in rf.get("due_diligence_checklist", [])] or ["- N/A"]
    return lines


def _sensitivity_lines(metrics: dict[str, Any]) -> list[str]:
    sens = metrics.get("sensitivity", {}) or {}
    if not sens:
        return []
    lines = ["", "### Stress Scenarios (deterministic)", "| Scenario | ROI % | Levered annual cash flow |", "|---|---|---|"]
    for name, sc in sens.items():
        lines.append(f"| {sc.get('description', name)} | {sc.get('roi_pct')} | {_fmt_inr(sc.get('annual_cash_flow_inr'))} |")
    return lines


def _citation_lines(report: dict[str, Any]) -> list[str]:
    rf = report.get("research_findings", {}) or {}
    evidence = rf.get("evidence", [])
    if not evidence:
        return [f"- {s}" for s in report.get("evidence_sources", [])] or ["- None"]
    lines = []
    for e in evidence:
        mark = "cited" if e.get("cited") else "retrieved"
        url = f" <{e['source_url']}>" if e.get("source_url") else ""
        lines.append(f"- **{e['evidence_id']}** ({mark}) {e['citation']}{url}")
    return lines


def _render_markdown(report: dict[str, Any], generated_at: str) -> str:
    rec = report.get("recommendation", {})
    metrics = report.get("investment_metrics", {})
    risk = report.get("risk_assessment", {})
    guardrail = report.get("guardrail_result", {})
    human = report.get("human_decision", {})
    property_data = report.get("property_data", {})
    market_data = report.get("market_data", {})

    lines = [
        f"# {PROJECT_TITLE}",
        "",
        f"**Generated:** {generated_at}",
        "",
        f"> **Data notice:** {report.get('data_disclaimer', 'Market and property data are simulated.')}",
        "",
        "## Request",
        f"- **Property address:** {report.get('property_address')}",
        f"- **Budget:** {_fmt_inr(report.get('budget_inr'))}",
        f"- **Investment strategy:** {report.get('investment_strategy')}",
        f"- **Investment horizon:** {report.get('investment_horizon_years')} years",
        "",
        "## Property Summary",
        f"- City / Locality: {property_data.get('city', 'N/A')} / {property_data.get('locality', 'N/A')}",
        f"- Property type: {property_data.get('property_type', 'N/A')}",
        f"- Price: {_fmt_inr(property_data.get('price_inr', 'N/A'))}",
        f"- Area: {property_data.get('area_sqft', 'N/A')} sq ft",
        f"- Amenities: {', '.join(property_data.get('amenities', [])) or 'N/A'}",
        "",
        "## Market Summary",
        f"- 5yr appreciation rate: {market_data.get('appreciation_rate_5yr_pct', 'N/A')}%",
        f"- Demand index: {market_data.get('demand_index', 'N/A')} / Supply index: {market_data.get('supply_index', 'N/A')}",
        f"- Avg rental yield: {market_data.get('rental_yield_avg_pct', 'N/A')}%",
        f"- RERA-registered projects: {market_data.get('rera_registered_projects_pct', 'N/A')}%",
        "" if market_data else "_No market data was available for this locality._",
        "",
        "## Financial Metrics",
        f"- ROI ({metrics.get('horizon_years', 'N/A')}yr): {metrics.get('roi_pct', 'N/A')}%",
        f"- Rental yield: {metrics.get('rental_yield_pct', 'N/A')}%",
        f"- Cap rate: {metrics.get('cap_rate_pct', 'N/A')}%",
        f"- Annual cash flow: {_fmt_inr(metrics.get('annual_cash_flow_inr', 'N/A'))} ({metrics.get('cash_flow_severity', 'N/A')})",
        f"- Break-even (years): {metrics.get('break_even_years', 'N/A')}",
        f"- Strong appreciation evidence: {metrics.get('strong_appreciation_evidence', 'N/A')}",
        *_sensitivity_lines(metrics),
        "",
        "### Financial Assumptions",
        *([f"- {a}" for a in metrics.get("assumptions", [])] or ["- N/A"]),
        "",
        "## Risk Assessment",
        f"- Risk score: {risk.get('risk_score', 'N/A')} / 100",
        f"- Data quality confidence: {risk.get('data_quality_confidence', 'N/A')}",
        f"- Factors: {json.dumps(risk.get('factors', {}))}",
        "",
        "## Recommendation",
        f"### Decision: {rec.get('decision', 'N/A')}",
        f"**Confidence:** {rec.get('confidence_score', 'N/A')}",
        "",
        f"{rec.get('justification', '')}",
        "",
        "### Supporting Evidence",
        *([f"- {e}" for e in rec.get("supporting_evidence", [])] or ["- None provided"]),
        "",
        "### Key Risks",
        *([f"- {e}" for e in rec.get("key_risks", [])] or ["- None stated"]),
        "",
        "### Analyst Assumptions",
        *([f"- {e}" for e in rec.get("assumptions", [])] or ["- None stated"]),
        "",
        *_research_lines(report),
        "",
        "## Guardrail Result",
        f"- Status: {guardrail.get('status', 'N/A')}",
        "### Reasons",
        *([f"- {r}" for r in guardrail.get("reasons", [])] or ["- None flagged"]),
        "",
        "## Sources and Citations",
        *_citation_lines(report),
        "",
        "## Data Provenance",
        *([f"- **{k}**: {v}" for k, v in (report.get("data_provenance") or {}).items()] or ["- N/A"]),
        "",
        "## Human Approval Decision",
        f"- Approved: {human.get('approved', 'N/A')}",
        f"- Feedback: {human.get('feedback') or 'N/A'}",
        "",
        f"_Report generated by Property Alpha AI on {generated_at}._",
    ]
    return "\n".join(lines)


def _write_markdown(report: dict[str, Any], path: str, generated_at: str) -> None:
    with open(path, "w") as f:
        f.write(_render_markdown(report, generated_at))


# --------------------------------------------------------------------------
# PDF (ReportLab)
# --------------------------------------------------------------------------


def _kv_table(rows: list[tuple[str, str]]) -> Table:
    table = Table(rows, colWidths=[5.5 * cm, 10.5 * cm])
    table.setStyle(
        TableStyle(
            [
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.HexColor("#dddddd")),
            ]
        )
    )
    return table


def _esc(text: Any) -> str:
    """Escape text for ReportLab's mini-markup (legal text contains '<', '&')."""
    from xml.sax.saxutils import escape

    return escape(str(text))


def _bullets(items: list[str], style, empty: str) -> list:
    return [Paragraph(_esc("• " + i), style) for i in items] or [Paragraph(_esc(empty), style)]


def _write_pdf(report: dict[str, Any], path: str, generated_at: str) -> None:
    rec = report.get("recommendation", {})
    metrics = report.get("investment_metrics", {})
    risk = report.get("risk_assessment", {})
    guardrail = report.get("guardrail_result", {})
    human = report.get("human_decision", {})
    property_data = report.get("property_data", {})
    market_data = report.get("market_data", {})

    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontSize=16, spaceAfter=10)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=12, spaceBefore=12, spaceAfter=6)
    body = ParagraphStyle("body", parent=styles["BodyText"], fontSize=10, leading=14)
    decision = rec.get("decision", "N/A")
    decision_style = ParagraphStyle(
        "decision",
        parent=styles["Heading2"],
        fontSize=14,
        textColor=DECISION_COLORS.get(decision, colors.black),
    )

    doc = SimpleDocTemplate(path, pagesize=A4, topMargin=1.5 * cm, bottomMargin=1.5 * cm)
    elements: list[Any] = [
        Paragraph(PROJECT_TITLE, h1),
        Paragraph(f"Generated: {generated_at}", body),
        Spacer(1, 4),
        Paragraph("<b>Data notice:</b> " + _esc(report.get("data_disclaimer", "Market and property data are simulated.")), body),
        Spacer(1, 8),
        _kv_table(
            [
                ("Property address", str(report.get("property_address", "N/A"))),
                ("Budget", _fmt_inr(report.get("budget_inr"))),
                ("Investment strategy", str(report.get("investment_strategy", "N/A"))),
                ("Investment horizon", f"{report.get('investment_horizon_years', 'N/A')} years"),
            ]
        ),
        Paragraph("Property Summary", h2),
        _kv_table(
            [
                ("City / Locality", f"{property_data.get('city', 'N/A')} / {property_data.get('locality', 'N/A')}"),
                ("Property type", str(property_data.get("property_type", "N/A"))),
                ("Price", _fmt_inr(property_data.get("price_inr", "N/A"))),
                ("Area", f"{property_data.get('area_sqft', 'N/A')} sq ft"),
                ("Amenities", ", ".join(property_data.get("amenities", [])) or "N/A"),
            ]
        ),
        Paragraph("Market Summary", h2),
        _kv_table(
            [
                ("5yr appreciation rate", f"{market_data.get('appreciation_rate_5yr_pct', 'N/A')}%"),
                ("Demand / Supply index", f"{market_data.get('demand_index', 'N/A')} / {market_data.get('supply_index', 'N/A')}"),
                ("Avg rental yield", f"{market_data.get('rental_yield_avg_pct', 'N/A')}%"),
                ("RERA-registered projects", f"{market_data.get('rera_registered_projects_pct', 'N/A')}%"),
            ]
            if market_data
            else [("Market data", "Not available for this locality")]
        ),
        Paragraph("Financial Metrics", h2),
        _kv_table(
            [
                ("ROI", f"{metrics.get('roi_pct', 'N/A')}% over {metrics.get('horizon_years', 'N/A')} yrs"),
                ("Rental yield", f"{metrics.get('rental_yield_pct', 'N/A')}%"),
                ("Cap rate", f"{metrics.get('cap_rate_pct', 'N/A')}%"),
                ("Annual cash flow", f"{_fmt_inr(metrics.get('annual_cash_flow_inr', 'N/A'))} ({metrics.get('cash_flow_severity', 'N/A')})"),
                ("Break-even", f"{metrics.get('break_even_years', 'N/A')} years"),
                ("Strong appreciation evidence", str(metrics.get("strong_appreciation_evidence", "N/A"))),
            ]
        ),
        Paragraph("Risk Assessment", h2),
        _kv_table(
            [
                ("Risk score", f"{risk.get('risk_score', 'N/A')} / 100"),
                ("Data quality confidence", str(risk.get("data_quality_confidence", "N/A"))),
                ("Factors", json.dumps(risk.get("factors", {}))),
            ]
        ),
        Paragraph("Recommendation", h2),
        Paragraph(f"Decision: {decision}", decision_style),
        Paragraph(f"Confidence: {rec.get('confidence_score', 'N/A')}", body),
        Spacer(1, 4),
        Paragraph(_esc(rec.get("justification", "")), body),
        Spacer(1, 6),
        Paragraph(
            _esc("Supporting evidence: " + ("; ".join(rec.get("supporting_evidence", [])) or "None provided")),
            body,
        ),
        Paragraph("Guardrail Result", h2),
        Paragraph(f"Status: {guardrail.get('status', 'N/A')}", body),
        Paragraph(_esc("Reasons: " + ("; ".join(guardrail.get("reasons", [])) or "None flagged")), body),
        Paragraph("Key Risks", h2),
        *_bullets(rec.get("key_risks", []), body, "None stated"),
        Paragraph("Financial Assumptions", h2),
        *_bullets(metrics.get("assumptions", []), body, "N/A"),
        Paragraph("Due Diligence Findings", h2),
        *[Paragraph(_esc(line.lstrip("#- ").strip()), body) for line in _research_lines(report)[1:] if line.strip()],
        Paragraph("Sources and Citations", h2),
        *[Paragraph(_esc(line.lstrip("- ").replace("**", "")), body) for line in _citation_lines(report)],
        Paragraph("Human Approval Decision", h2),
        _kv_table(
            [
                ("Approved", str(human.get("approved", "N/A"))),
                ("Feedback", human.get("feedback") or "N/A"),
                ("Timestamp", generated_at),
            ]
        ),
    ]
    doc.build(elements)


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------


def generate_reports(
    report: dict[str, Any], output_dir: str = "reports", base_name: str | None = None
) -> dict[str, str]:
    """Generate report.json / report.md / report.pdf (or <base_name>.* if
    given) under output_dir. Only valid for an approved final report."""
    if report.get("status") != "approved":
        raise ValueError("Refusing to generate a report for a recommendation that was not human-approved.")

    import os

    os.makedirs(output_dir, exist_ok=True)
    name = base_name or slugify(str(report.get("property_address", "report")))
    generated_at = _timestamp()

    paths = {
        "json": os.path.join(output_dir, f"{name}.json"),
        "md": os.path.join(output_dir, f"{name}.md"),
        "pdf": os.path.join(output_dir, f"{name}.pdf"),
    }

    _write_json(report, paths["json"], generated_at)
    _write_markdown(report, paths["md"], generated_at)
    _write_pdf(report, paths["pdf"], generated_at)

    return paths
