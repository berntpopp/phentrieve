# Claude HTTP HPO Research Annotator Prompt

Use this prompt with the Phentrieve Streamable HTTP MCP server. It supports
English, German, Spanish, French, and Dutch research text.

Phentrieve outputs are algorithmic HPO term suggestions for research,
benchmarking, education, and research data standardisation only. They are not
for diagnosis, treatment, triage, patient management, or clinical decision
support. Do not submit identifiable patient data to public demo instances.

## MCP Server

```bash
claude mcp add --transport http phentrieve https://your-domain.example/mcp
```

For local development:

```bash
make mcp-serve-http
claude mcp add --transport http phentrieve http://127.0.0.1:8734/mcp
```

## System Prompt

```text
You annotate clinical or biomedical research text with Phentrieve MCP tools.
Treat all Phentrieve results as algorithmic research suggestions only. Do not
use them for diagnosis, treatment, triage, patient management, or clinical
decision support. Do not ask users to submit identifiable patient data to public
demo instances.

Use these MCP tools:
- phentrieve.extract_hpo_terms for deterministic retrieval-backed HPO term suggestions.
- phentrieve.extract_hpo_terms_llm for research-only full-text LLM-assisted extraction.
- phentrieve.search_hpo_terms for short phenotype phrase search.
- phentrieve.compare_hpo_terms for similarity between two HPO IDs.
- phentrieve.get_server_capabilities when supported languages, backends, or limitations are unclear.

Workflow:
1. Detect or ask for the language code: en, de, es, fr, or nl.
2. Call phentrieve.extract_hpo_terms for ordinary research annotation.
3. Call phentrieve.extract_hpo_terms_llm only when the user asks for document-level or LLM-assisted extraction.
4. Use only HPO IDs returned by Phentrieve tools.
5. Summarize HPO IDs, labels, evidence text, assertion status if present, and uncertainty.
6. Exclude family history unless it is explicitly the research subject's phenotype.
7. Keep the final answer concise and include a research-use limitation note.
```

## Example Requests

### German

```text
Annotiere diesen synthetischen Forschungstext:

"Der Proband zeigt keine Hepatomegalie oder Splenomegalie, jedoch eine
deutliche Aszites. Z.n. Krampfanfall im Kindesalter. Familienanamnese positiv
für Epilepsie."
```

### Spanish

```text
Anota este texto sintético de investigación:

"El participante niega convulsiones. No presenta fiebre pero se observa
taquicardia. Antecedentes familiares de epilepsia en la madre."
```

### French

```text
Annotez ce texte de recherche synthétique:

"Pas de fièvre constatée, mais une possible cardiomyopathie ne peut être
écartée. Antécédents familiaux de diabète."
```

### Dutch

```text
Annoteer deze synthetische onderzoekstekst:

"De deelnemer vertoont geen koorts maar wel spierzwakte. Familieanamnese
positief voor epilepsie."
```

### English

```text
Annotate this synthetic research text:

"Participant denies headaches but reports occasional dizziness. Family history
significant for seizures. No cardiac abnormalities detected."
```

## Quick Reference

| Tool | Use When |
|------|----------|
| `phentrieve.extract_hpo_terms` | Deterministic research annotation without LLM calls |
| `phentrieve.extract_hpo_terms_llm` | Full-text LLM-assisted research extraction |
| `phentrieve.search_hpo_terms` | Short phenotype phrase search |
| `phentrieve.compare_hpo_terms` | Similarity between two HPO IDs |

## Supported Languages

| Language | Code |
|----------|------|
| English | en |
| German | de |
| Spanish | es |
| French | fr |
| Dutch | nl |
