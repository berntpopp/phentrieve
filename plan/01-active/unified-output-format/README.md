# Unified Output Format with Phenopackets Integration

**Status:** 📋 Planning Phase - Awaiting Approval
**Created:** 2025-01-21
**Priority:** High
**Estimated Effort:** 5 weeks

---

## 📁 Contents

This folder contains comprehensive planning documentation for implementing a unified output format architecture using GA4GH Phenopackets v2.0:

### 📄 Core Documents

1. **[GitHub Issue #87](https://github.com/berntpopp/phentrieve/issues/87)** ⭐ **START HERE**
   - Concise implementation plan
   - Success criteria and timeline
   - 5-week phased roadmap (3 weeks MVP + 2 weeks polish)
   - **Read this first for executive summary**

2. **[UNIFIED-OUTPUT-FORMAT-PHENOPACKETS.md](./UNIFIED-OUTPUT-FORMAT-PHENOPACKETS.md)**
   - Comprehensive architectural design (89 KB, 13 sections)
   - Deep dive into Phenopackets capabilities and limitations
   - Detailed code examples and data structures
   - Alternative architectures considered
   - Research findings from web search
   - **Read this for technical deep dive**

3. **[UNIFIED-OUTPUT-FORMAT-PHENOPACKETS-REVIEW.md](./UNIFIED-OUTPUT-FORMAT-PHENOPACKETS-REVIEW.md)**
   - Senior developer code review
   - SOLID/DRY/KISS/YAGNI analysis
   - Critical issues and recommendations
   - Simplified architecture proposal
   - Risk assessment and mitigation
   - **Read this for code quality perspective**

---

## 🎯 Quick Summary

### Problem
- Fragmented output formats between `query` and `text process` commands
- No clinical interoperability (GA4GH standards)
- Lost provenance (can't trace HPO terms to source text)
- Cannot save/load analysis state for editing

### Solution
Unified three-layer architecture:
1. **Internal**: `PhentrieveAnalysis` wrapper with embedded Phenopacket
2. **Converters**: Pluggable exporters (JSON, Phenopacket, CSV)
3. **Exports**: User-selectable formats

### Key Improvements
- ✅ Single canonical data format (no duplication)
- ✅ GA4GH Phenopackets v2.0 compliance
- ✅ Full metadata preservation (models, configs, timestamps)
- ✅ Flexible export (JSON, CSV, Phenopacket)
- ✅ 72% less code than original plan (2400 LOC vs 8500 LOC)

---

## 📊 Document Comparison

| Document | Length | Audience | Purpose |
|----------|--------|----------|---------|
| GitHub Issue #87 | Concise | Project team | Actionable implementation plan |
| UNIFIED-OUTPUT-FORMAT-PHENOPACKETS.md | 89 KB | Architects | Comprehensive technical design |
| UNIFIED-OUTPUT-FORMAT-PHENOPACKETS-REVIEW.md | 26 KB | Developers | Code quality analysis |

---

## 🚀 Reading Order

### For Project Managers / Stakeholders
1. ✅ Start with **[GitHub Issue #87](https://github.com/berntpopp/phentrieve/issues/87)** (5 min read)
   - Quick overview of problem, solution, timeline
   - Success criteria and deliverables
2. ✅ Skim **REVIEW.md** § Executive Summary (2 min read)
   - Understand complexity reduction and risks

### For Technical Leads / Architects
1. ✅ Start with **[GitHub Issue #87](https://github.com/berntpopp/phentrieve/issues/87)** (5 min read)
2. ✅ Read **REVIEW.md** in full (20 min read)
   - Understand code quality concerns and recommendations
3. ✅ Read **PLAN.md** § 3 (Architecture Design) (30 min read)
   - Deep dive into technical architecture

### For Developers (Implementation Team)
1. ✅ Start with **[GitHub Issue #87](https://github.com/berntpopp/phentrieve/issues/87)** (5 min read)
   - Understand success criteria and tasks
2. ✅ Read **REVIEW.md** § Critical Issues and Alternative Architecture (15 min read)
   - Understand simplified design patterns
3. ✅ Reference **PLAN.md** § 3.2 (Core Data Structures) as needed (20 min read)
   - Code examples and implementation details

---

## 📋 Implementation Status

### Phase 1: MVP (3 weeks)
- [ ] **Week 1**: Core data structures
- [ ] **Week 2**: Essential converters (JSON, Phenopacket, CSV)
- [ ] **Week 3**: CLI integration

### Phase 2: Production-Ready (2 weeks)
- [ ] **Week 4**: Testing & validation
- [ ] **Week 5**: Documentation & polish

### Phase 3: Extensions (Future)
- ⏸️ Interactive mode (deferred)
- ⏸️ Additional exporters (TSV, TXT, JSONL)
- ⏸️ API integration

---

## 🔑 Key Decisions

### ✅ Approved Design Choices
1. **Computed Properties**: Eliminate data duplication via `@cached_property`
2. **Three Exporters**: Start with JSON, Phenopacket, CSV (not 8)
3. **One-Way Conversion**: Phenopacket export only (no import)
4. **Pydantic Validation**: Automatic validation on data structure creation
5. **Metadata Grouping**: Split into config/stats/provenance

### ⏸️ Deferred to Phase 3
1. Interactive TUI mode
2. TSV, TXT, JSONL exporters
3. Plugin architecture for custom exporters

### ❌ Rejected from Scope
1. Bidirectional Phenopacket conversion (lossy by design)
2. Legacy format converters (use adapters instead)
3. Multiple confidence score strategies (keep one)

---

## 📈 Metrics

### Code Reduction (Original Plan → Simplified)
- **Total LOC**: 8500 → 2400 (72% reduction)
- **Converters**: 8 → 3 (63% reduction)
- **Timeline**: 10 weeks → 5 weeks (50% faster)
- **Complexity**: 7/10 → 4/10 (43% reduction)

### Quality Targets
- **Test Coverage**: >90% for new code
- **Type Safety**: 0 mypy errors
- **Linting**: 0 Ruff errors
- **Performance**: <10% regression

---

## 🔗 Related Resources

### External Documentation
- [GA4GH Phenopackets v2.0 Docs](https://phenopacket-schema.readthedocs.io/)
- [Phenopackets Python Library](https://github.com/phenopackets/phenopacket-schema)
- [HPO Browser](https://hpo.jax.org/)
- [Evidence & Conclusion Ontology (ECO)](http://www.evidenceontology.org/)

### Internal References
- `plan/STATUS.md` - Overall project status
- `plan/02-completed/TESTING-MODERNIZATION-PLAN.md` - Testing framework
- `phentrieve/retrieval/output_formatters.py` - Current output formatters (to be replaced)
- `api/schemas/` - Current API schemas (to be updated)

---

## ❓ Open Questions

### For Stakeholders
1. Should Phenopackets include patient IDs? (GDPR/HIPAA implications)
2. Should we capture clinical context (document type, clinician, encounter date)?
3. What should be the default output format for CLI?

### For Technical Team
1. Use dataclasses or Pydantic models? (Review suggests Pydantic)
2. When to validate Phenopackets? (On export or on-demand?)
3. Database storage for analyses? (File-based or PostgreSQL/MongoDB?)

---

## 🎯 Next Actions

### Immediate (This Week)
1. ⏳ Stakeholder review of IMPLEMENTATION-ISSUE.md
2. ⏳ Technical team review of REVIEW.md recommendations
3. ⏳ Decide on open questions (patient IDs, clinical context, default format)

### Week 1 (Upon Approval)
1. ⏳ Create feature branch `feature/unified-output-format`
2. ⏳ Implement core data structures
3. ⏳ Set up test framework for new code

---

**Status**: 📋 Awaiting stakeholder approval
**Last Updated**: 2025-01-21
**Next Review**: Upon feedback from stakeholders and technical team
