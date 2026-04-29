# Privacy and LLM Processing

This page describes the intended privacy posture for a public Phentrieve
research service, including deployments such as `phentrieve.kidney-genetics.org`.

Phentrieve can process text through two backend modes:

- **Standard extraction** uses local retrieval and text-processing components.
- **LLM extraction** may send submitted text to the external LLM provider
  configured by the service operator.

## Public Hosted Mode

Public deployments should enable:

```bash
PHENTRIEVE_PUBLIC_HOSTED_MODE=true
PHENTRIEVE_REQUIRE_RESEARCH_ACK=true
```

This keeps LLM functionality available while requiring clients to acknowledge
the research-use limitation before text-bearing requests are accepted.

## External LLM Disclosure

When LLM extraction is enabled, the operator should disclose:

- which LLM provider or provider category is used;
- that submitted text may be transmitted to that provider for processing;
- whether provider-side logging, retention, or model-training use is disabled;
- whether text is stored by the Phentrieve service itself;
- how long application logs are retained;
- who to contact for privacy or data-removal questions.

The frontend displays an LLM-specific notice when the LLM extraction backend is
selected. Operators should keep the deployment privacy notice consistent with
the configured LLM provider and data processing terms.

## User Data Guidance

Users should submit research text only. They should not submit names, contact
details, record numbers, addresses, dates that directly identify a person, or
other directly identifying information.

Research text may still contain health-related information. Operators remain
responsible for assessing whether their deployment has an appropriate legal
basis, privacy notice, data processing agreement, cross-border transfer basis,
and security controls for the data they accept.

## Logging and Storage

The public research service should avoid logging raw submitted text. Logs should
prefer metadata such as request size, selected backend, status code, latency,
and quota state.

Phentrieve does not need to store submitted text for normal query and text
processing requests. Phenopacket exports intentionally include source text in
the returned artifact when the user chooses to export it; exported artifacts are
the user's responsibility after download or downstream transfer.

## Operator Checklist

Before enabling a public hosted research deployment with LLM extraction:

- publish a privacy notice that names the service operator;
- document the configured LLM provider and data handling terms;
- keep raw-text logging disabled;
- require the research-use acknowledgement;
- keep clinical decision-making claims out of the UI and documentation;
- provide a contact route for privacy and security reports;
- review whether GDPR, institutional review, data processing, or transfer
  requirements apply to the deployment.
