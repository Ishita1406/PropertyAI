"""AGENT 1 — Property Research & Due Diligence Agent (LLM, Groq).

What makes this an agent rather than a renamed vector search:

1. **Planning with tools.** Given the property profile (simulated data), the
   LLM decides *which* legal/policy questions matter for this property and
   strategy and issues its own searches through two tools — one over official
   documents (filterable by document type), one over simulated market notes —
   for up to ``RESEARCH_MAX_TOOL_ROUNDS`` rounds.
2. **Synthesis.** A second, structured call turns the retrieved evidence into
   regulatory requirements, property concerns, evidence gaps, conflicts and a
   due-diligence checklist, citing evidence IDs.
3. **Deterministic post-validation.** Citations not in the retrieved set are
   stripped; uncited requirements are moved to ``unsupported_statements``;
   binding status is taken from document metadata (never from the LLM);
   structural gaps (missing documents, missing simulated data, State-specific
   rules) are always added.

If the LLM is unavailable after bounded retries, the agent degrades to a fixed
baseline retrieval and returns ``status="degraded_llm_unavailable"`` — the
guardrail then forces human review.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from property_advisor.agents._llm_utils import invoke_with_retries
from property_advisor.config import RESEARCH_MAX_TOOL_ROUNDS, get_llm
from property_advisor.logging_utils import timed_node
from property_advisor.rag.catalog import MISSING_DOCUMENTS, OFFICIAL_DOCUMENTS
from property_advisor.schemas import ResearchFindingsOutput
from property_advisor.state import PropertyState
from property_advisor.tools.rag_tool import format_citation, query_rag
from property_advisor.validation import detect_corpus_conflicts

logger = logging.getLogger("property_advisor")

HITS_PER_SEARCH = 4
MAX_CALLS_PER_ROUND = 4
MAX_EVIDENCE = 24
SNIPPET_CHARS = 420
OFFICIAL_SOURCE_TYPES = ["official_government", "secondary"]
DOCUMENT_TYPES = (
    "primary_legislation",
    "model_legislation",
    "government_guidelines",
    "government_research",
    "secondary_commentary",
)

PLANNER_PROMPT = """You are the Property Research & Due Diligence Agent for Indian residential real estate.
Your job now: decide what evidence to retrieve for THIS property, then retrieve it with the tools.

Think about which legal and policy questions actually matter given the property type, \
location, investment strategy and the risk/market signals provided. Examples:
- buying in a new/under-construction project -> RERA registration, disclosures, delay remedies;
- any purchase -> compulsory registration of sale deeds/leases (Registration Act);
- rental strategy -> tenancy agreements, rent/security deposit, eviction (Model Tenancy Act — a MODEL law);
- access, light, drainage rights -> Indian Easements Act;
- builder deficiency/unfair practice -> Consumer Protection Act;
- FSI, setbacks, flood/structural safety -> Model Building Bye-Laws / URDPFI zoning guidance;
- price index context -> NHB RESIDEX methodology;
- locality sentiment -> search_market_commentary (SIMULATED notes only).

Rules: issue 3-6 focused searches in legal/policy wording, at most 4 per turn. \
Do not write an analysis yet. When you have enough evidence reply with the single word DONE."""

SYNTHESIS_PROMPT = """You are the Property Research & Due Diligence Agent. Produce structured findings \
from the evidence list ONLY.

Hard rules:
- Cite evidence by its ID (E1, E2, ...). Never invent IDs, section numbers, page numbers or documents.
- Property, price, rent, market and risk figures are SIMULATED project data — call them simulated; \
never present them as real market observations. Do not compute any financial figures.
- source_type 'simulated' evidence is synthetic commentary, NOT a real report or law.
- 'secondary_commentary' is not authoritative law; prefer primary legislation when both exist.
- model_legislation (Model Tenancy Act, Model Building Bye-Laws) is NOT binding unless the State/UT adopted it; say so.
- Central Acts do not automatically override State/local rules; RERA rules, stamp duty, registration fees \
and building bye-laws are State/local — flag them as gaps when relevant.
- Report missing, outdated or conflicting evidence explicitly.
Be concise."""


class EvidenceRegistry:
    """Assigns stable evidence IDs (E1, E2, ...) to retrieved chunks."""

    def __init__(self) -> None:
        self._by_chunk: dict[str, dict[str, Any]] = {}
        self.order: list[str] = []
        self.searches: list[dict[str, Any]] = []

    def add(self, hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
        added = []
        for h in hits:
            cid = h.get("chunk_id") or f"{h.get('source')}::{hash(h.get('text', ''))}"
            if cid not in self._by_chunk:
                if len(self.order) >= MAX_EVIDENCE:
                    continue
                eid = f"E{len(self.order) + 1}"
                self._by_chunk[cid] = {**h, "evidence_id": eid, "chunk_id": cid}
                self.order.append(cid)
            added.append(self._by_chunk[cid])
        return added

    def all(self) -> list[dict[str, Any]]:
        return [self._by_chunk[c] for c in self.order]

    def ids(self) -> set[str]:
        return {self._by_chunk[c]["evidence_id"] for c in self.order}

    def get(self, evidence_id: str) -> dict[str, Any] | None:
        for c in self.order:
            if self._by_chunk[c]["evidence_id"] == evidence_id:
                return self._by_chunk[c]
        return None


def _describe_hit(h: dict[str, Any], snippet_chars: int = SNIPPET_CHARS) -> str:
    text = " ".join((h.get("text") or "").split())[:snippet_chars]
    ocr = " (OCR text)" if h.get("extraction_method") == "ocr" else ""
    return (
        f"{h['evidence_id']} | {format_citation(h)} | binding_status={h.get('binding_status', 'unknown')}{ocr}\n"
        f"   \"{text}\""
    )


def _search(registry: EvidenceRegistry, query: str, how: str, *, simulated: bool, document_type: str | None = None) -> str:
    try:
        if simulated:
            hits = query_rag(query, k=HITS_PER_SEARCH, source_type="simulated")
        else:
            dt = document_type if document_type in DOCUMENT_TYPES else None
            hits = query_rag(query, k=HITS_PER_SEARCH, source_type=OFFICIAL_SOURCE_TYPES, document_type=dt)
    except Exception as exc:  # noqa: BLE001
        registry.searches.append({"query": query, "tool": how, "error": str(exc)})
        return f"Search failed: {exc}"
    added = registry.add(hits)
    registry.searches.append(
        {"query": query, "tool": how, "document_type": document_type, "evidence_ids": [h["evidence_id"] for h in added]}
    )
    if not added:
        return "No results."
    return "\n".join(_describe_hit(h, 300) for h in added)


def _make_tools(registry: EvidenceRegistry):
    @tool
    def search_official_documents(query: str, document_type: str = "any") -> str:
        """Search official Indian legislation and government publications in the knowledge base.
        document_type: one of any, primary_legislation, model_legislation, government_guidelines,
        government_research, secondary_commentary."""
        return _search(registry, query, "search_official_documents", simulated=False, document_type=document_type)

    @tool
    def search_market_commentary(query: str) -> str:
        """Search SIMULATED locality/market commentary notes (synthetic project data, not real reports)."""
        return _search(registry, query, "search_market_commentary", simulated=True)

    return [search_official_documents, search_market_commentary]


def _property_context(state: PropertyState) -> str:
    p = state.property_data
    metrics = state.investment_metrics
    risk = {k: v for k, v in state.risk_assessment.items() if k in ("risk_score", "factors", "raw_indicators")}
    profile = {
        "address_as_entered": state.property_address,
        "strategy": state.investment_strategy,
        "horizon_years": state.investment_horizon_years,
        "property (SIMULATED)": {k: p.get(k) for k in ("city", "locality", "property_type", "price_inr", "area_sqft", "amenities", "location_attributes") if k in p} or "NOT FOUND in simulated dataset",
        "market (SIMULATED)": state.market_data or "NOT AVAILABLE for this locality",
        "risk (SIMULATED indicators)": risk,
        "deterministic metrics": {k: metrics.get(k) for k in ("rental_yield_pct", "roi_pct", "cash_flow_severity", "levered_cash_flow_negative") if k in metrics},
    }
    return json.dumps(profile, default=str)


def _catalogue_note() -> str:
    excluded = [f"{m['document_title']}: {m.get('exclusion_reason', '')[:140]}" for m in OFFICIAL_DOCUMENTS.values() if not m["include_in_rag"]]
    missing = [d["document_title"] for d in MISSING_DOCUMENTS]
    return json.dumps({"not_in_knowledge_base": missing, "excluded_as_outdated": excluded})


def _baseline_queries(state: PropertyState) -> list[tuple[str, bool, str | None]]:
    """Deterministic retrieval used for the locality commentary and as the
    fallback when the LLM is unavailable: (query, simulated?, document_type)."""
    locality = state.property_data.get("locality", "")
    city = state.property_data.get("city", "")
    queries: list[tuple[str, bool, str | None]] = [
        (f"{locality} {city} property market outlook risks".strip(), True, None),
        ("registration of real estate project with Real Estate Regulatory Authority before advertising or selling", False, "primary_legislation"),
        ("documents of which registration is compulsory sale of immovable property lease", False, "primary_legislation"),
    ]
    if state.investment_strategy == "rental":
        queries.append(("tenancy agreement security deposit rent revision eviction", False, "model_legislation"))
    if state.risk_assessment.get("factors", {}).get("flood_risk", 0) >= 40:
        queries.append(("flood prone area building safety drainage plinth level", False, None))
    return queries


def _deterministic_gaps(state: PropertyState) -> list[dict[str, str]]:
    gaps = []
    if not state.property_data:
        gaps.append({"gap": "Property not found in the simulated property dataset.", "why_it_matters": "No property attributes to evaluate.", "origin": "deterministic"})
    if not state.market_data:
        gaps.append({"gap": "No simulated market trend data for this locality.", "why_it_matters": "Rent and appreciation inputs default to zero; metrics are unreliable.", "origin": "deterministic"})
    if not state.risk_assessment.get("raw_indicators"):
        gaps.append({"gap": "No simulated risk indicators for this locality.", "why_it_matters": "Risk score defaults to a moderate 50 with zero data confidence.", "origin": "deterministic"})
    city = state.property_data.get("city", "the relevant State")
    gaps.append({
        "gap": f"State RERA rules, State stamp duty schedule, registration fees and local building bye-laws/DCR for {city} are not in the knowledge base.",
        "why_it_matters": "These State/local instruments, not central model texts, govern the actual transaction.",
        "origin": "deterministic",
    })
    for d in MISSING_DOCUMENTS:
        gaps.append({"gap": f"Missing source: {d['document_title']}.", "why_it_matters": d["impact"], "origin": "deterministic", "official_source_url": d["official_source_url"]})
    return gaps


def _validate_ids(ids: list[str], registry: EvidenceRegistry, bad: list[str]) -> list[str]:
    valid = []
    for i in ids:
        i = i.strip()
        if registry.get(i):
            valid.append(i)
        else:
            bad.append(i)
    return list(dict.fromkeys(valid))


def assemble_findings(
    llm_out: ResearchFindingsOutput | None,
    registry: EvidenceRegistry,
    state: PropertyState,
    status: str,
    errors: list[str],
) -> dict[str, Any]:
    """Deterministically validate/enrich the LLM's findings."""
    unverifiable: list[str] = []
    requirements, unsupported, concerns, conflicts = [], [], [], []
    llm_gaps, checklist, summary = [], [], ""

    if llm_out is not None:
        summary = llm_out.summary
        checklist = llm_out.due_diligence_checklist[:6]
        for r in llm_out.regulatory_requirements[:6]:
            ids = _validate_ids(r.citation_ids, registry, unverifiable)
            if not ids:
                unsupported.append({"statement": r.requirement, "reason": "no valid citation to retrieved evidence"})
                continue
            cited = [registry.get(i) for i in ids]
            binding = sorted({c.get("binding_status", "unknown") for c in cited})
            authoritative = any(c.get("source_type") == "official_government" for c in cited)
            requirements.append({
                "requirement": r.requirement,
                "relevance": r.relevance,
                "citation_ids": ids,
                "citations": [format_citation(c) for c in cited],
                "binding_status": binding,
                "authoritative_source": authoritative,
            })
        for c in llm_out.concerns[:6]:
            ids = _validate_ids(c.citation_ids, registry, unverifiable)
            item = {"concern": c.concern, "severity": c.severity, "basis": c.basis, "citation_ids": ids}
            if c.basis in ("official_document", "both") and not ids:
                item["citation_status"] = "uncited — treat as unsupported"
            concerns.append(item)
        llm_gaps = [{"gap": g.gap, "why_it_matters": g.why_it_matters, "origin": "agent"} for g in llm_out.evidence_gaps[:5]]
        conflicts = list(llm_out.conflicting_evidence[:3])

    # Deterministic conflict detection on the simulated notes (kept from the original guardrail).
    for c in detect_corpus_conflicts(h.get("source", "") for h in registry.all()):
        conflicts.append(f"SIMULATED analyst notes disagree: {c}")

    cited_ids = {i for r in requirements for i in r["citation_ids"]} | {i for c in concerns for i in c["citation_ids"]}
    evidence = []
    for h in registry.all():
        evidence.append({
            "evidence_id": h["evidence_id"],
            "citation": format_citation(h),
            "document_id": h.get("document_id"),
            "document_title": h.get("document_title"),
            "publisher": h.get("publisher"),
            "source_url": h.get("source_url"),
            "document_type": h.get("document_type"),
            "source_type": h.get("source_type"),
            "binding_status": h.get("binding_status"),
            "jurisdiction": h.get("jurisdiction"),
            "section_number": h.get("section_number"),
            "page_number": h.get("page_number"),
            "page_end": h.get("page_end"),
            "effective_date": h.get("effective_date"),
            "version": h.get("version"),
            "extraction_method": h.get("extraction_method"),
            "excerpt": " ".join((h.get("text") or "").split())[:400],
            "cited": h["evidence_id"] in cited_ids,
        })

    if not any(e["source_type"] == "official_government" for e in evidence):
        llm_gaps.append({"gap": "No official legal/policy evidence was retrieved.", "why_it_matters": "Regulatory requirements cannot be cited.", "origin": "deterministic"})
    if any(e["extraction_method"] == "ocr" and e["cited"] for e in evidence):
        llm_gaps.append({"gap": "Some cited evidence comes from OCR text (URDPFI/scanned pages).", "why_it_matters": "OCR may contain recognition errors; verify against the original PDF page.", "origin": "deterministic"})

    return {
        "status": status,
        "summary": summary or "LLM research synthesis unavailable — only baseline retrieval was performed; findings below are deterministic.",
        "regulatory_requirements": requirements,
        "concerns": concerns,
        "evidence_gaps": llm_gaps + _deterministic_gaps(state),
        "conflicting_evidence": list(dict.fromkeys(conflicts)),
        "due_diligence_checklist": checklist,
        "unsupported_statements": unsupported,
        "unverifiable_citations": sorted(set(unverifiable)),
        "evidence": evidence,
        "retrieval_log": registry.searches,
        "data_provenance_note": "Property/market/risk inputs are SIMULATED; legal evidence is from official documents unless tagged secondary/simulated.",
        "errors": errors,
    }


def _run_planning_loop(llm, registry: EvidenceRegistry, context: str) -> None:
    tools = _make_tools(registry)
    tool_map = {t.name: t for t in tools}
    llm_tools = llm.bind_tools(tools)
    messages: list = [SystemMessage(PLANNER_PROMPT), HumanMessage(f"Property profile:\n{context}\n\nKnowledge-base limits: {_catalogue_note()}")]
    for _ in range(RESEARCH_MAX_TOOL_ROUNDS):
        ai = invoke_with_retries(lambda: llm_tools.invoke(messages), "research_agent.plan")
        messages.append(ai)
        calls = getattr(ai, "tool_calls", None) or []
        if not calls:
            break
        for n, tc in enumerate(calls):
            fn = tool_map.get(tc.get("name"))
            if fn is None:
                result = f"Unknown tool {tc.get('name')}"
            elif n >= MAX_CALLS_PER_ROUND:
                result = "Skipped: per-turn search limit reached."
            else:
                result = fn.invoke(tc.get("args", {}))
            messages.append(ToolMessage(content=result, tool_call_id=tc.get("id", f"call_{n}")))


@timed_node("research_agent")
def research_agent(state: PropertyState) -> dict:
    registry = EvidenceRegistry()
    errors: list[str] = []
    context = _property_context(state)
    llm_out: ResearchFindingsOutput | None = None
    status = "complete"

    # Locality commentary is always retrieved so conflicting simulated notes are
    # surfaced deterministically regardless of the LLM's plan.
    q, _, _ = _baseline_queries(state)[0]
    _search(registry, q, "baseline_locality_commentary", simulated=True)

    try:
        llm = get_llm(temperature=0.1, max_tokens=1500)
        _run_planning_loop(llm, registry, context)
        if not any(h.get("source_type") != "simulated" for h in registry.all()):
            # The model did not retrieve any official evidence: fall back to baseline queries.
            for query, sim, dt in _baseline_queries(state)[1:]:
                _search(registry, query, "baseline_fallback", simulated=sim, document_type=dt)
        evidence_text = "\n".join(_describe_hit(h) for h in registry.all()) or "(no evidence retrieved)"
        structured = llm.with_structured_output(ResearchFindingsOutput)
        prompt = (
            f"Property profile:\n{context}\n\nKnowledge-base limits: {_catalogue_note()}\n\n"
            f"Evidence list:\n{evidence_text}"
        )
        llm_out = invoke_with_retries(
            lambda: structured.invoke([SystemMessage(SYNTHESIS_PROMPT), HumanMessage(prompt)]),
            "research_agent.synthesise",
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("research_agent degraded: %s", exc)
        errors.append(f"Research agent LLM unavailable: {exc}")
        status = "degraded_llm_unavailable"
        for query, sim, dt in _baseline_queries(state)[1:]:
            _search(registry, query, "baseline_fallback", simulated=sim, document_type=dt)

    findings = assemble_findings(llm_out, registry, state, status, errors)
    rag_context = [
        {k: h.get(k) for k in ("evidence_id", "chunk_id", "source", "text", "document_title", "document_type", "source_type", "section_number", "page_number", "page_end")}
        for h in registry.all()
    ]
    update: dict[str, Any] = {"research_findings": findings, "rag_context": rag_context}
    if errors:
        update["errors"] = [*state.errors, *errors]
    return update
