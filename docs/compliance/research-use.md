# Research Use Only

Phentrieve is provided for research, education, benchmarking, and knowledge
discovery. It maps input text to Human Phenotype Ontology (HPO) terms and may
return incomplete, incorrect, or contextually inappropriate results.

Phentrieve is not a medical device and must not be used for diagnosis,
treatment selection, patient triage, emergency workflows, or any other clinical
decision-making. Human review and separate clinical validation are required
before any downstream clinical or operational use.

## EU AI Act Positioning

This project is intended to remain outside clinical high-risk use by keeping the
public service research-only and by avoiding claims that it performs diagnosis,
patient management, or therapeutic decision support.

Operators who deploy Phentrieve in a different context are responsible for their
own legal assessment. A deployment used for clinical decision support, patient
care workflows, or automated decision-making may require obligations that are
not covered by this research-use configuration.

## Public Service Guardrails

Text-bearing API endpoints can require explicit acknowledgement by setting:

```bash
PHENTRIEVE_PUBLIC_HOSTED_MODE=true
```

or:

```bash
PHENTRIEVE_REQUIRE_RESEARCH_ACK=true
```

Clients must then send:

```http
X-Phentrieve-Research-Use-Acknowledged: true
```

LLM extraction remains available in public-hosted research deployments. If an
external LLM provider is configured, operators should maintain a separate
privacy, data processing, logging, and vendor-transfer assessment and disclose
that submitted text may be processed by an external AI provider.

## Data Handling

Do not submit protected health information, directly identifying information, or
text that you are not authorized to process. The CLI prints a research-use
notice for text-bearing commands, the web app requires the current disclaimer to
be acknowledged, and the frontend avoids logging raw submitted text.

See [Privacy and LLM Processing](privacy-and-llm-processing.md) for the
recommended public-service posture when LLM extraction sends text to an external
AI provider.

Phenopacket exports include a Phentrieve intended-use metadata reference so
exported artifacts carry the research-use limitation with them.
