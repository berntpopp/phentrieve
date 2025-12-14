#!/usr/bin/env python3
"""
Simple test script for mention-level HPO extraction.
"""

import logging
from phentrieve.retrieval.dense_retriever import DenseRetriever
from phentrieve.text_processing.mention_extraction_orchestrator import orchestrate_mention_extraction
from phentrieve.embeddings import load_embedding_model

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Sample clinical text from ID_68 benchmark
    text = "An 8 years old girl who was born at full term via NSVD to G4P2+1 33 years old mother. A decrease in fetal movements was noted during pregnancy, and her birth weight was 2.8 kg. After birth, she was noted to be floppy with poor sucking. At one month of age, she started to have choking episodes that turned to be due to severe GERD, and Niessel fundoplication was performed. Aspiration problems persisted, and the fundoplication procedure was repeated later. She also has developed seizures and continues to seize daily despite multiple attempts to control her seizures with medications. Her seizures' semiology appears to be as opisthotonic posturing. She also has bronchial asthma. Her parents report that she barely can feel pain, and she has no tears when she cries even though she has normal sweating. She demonstrates a self-mutilating behavior by scratching her face. Parents are consanguineous, and there is no family history of a similar condition. Currently, she has a developmental delay affecting all domains, and her IQ was estimated to be 20-30. She can vocalize, but cannot talk. She can walk although her gait is unstable. On examination, she has poor growth with a significant microcephaly. She has subtle dysmorphia characterized as hypotelorism and tapering of fingers. MRI showed thinning of the corpus callosum and mild atrophy of the cerebellar vermis."

    logger.info("Testing mention-level HPO extraction...")

    # Load embedding model
    logger.info("Loading embedding model...")
    model = load_embedding_model("FremyCompany/BioLORD-2023-M")

    # Initialize retriever
    logger.info("Initializing retriever...")
    retriever = DenseRetriever.from_model_name(
        model=model,
        model_name="FremyCompany/BioLORD-2023-M"
    )
    if retriever is None:
        logger.error("Failed to initialize retriever. Make sure the index exists.")
        return

    # Run mention extraction
    logger.info("Running mention extraction...")
    result = orchestrate_mention_extraction(
        text=text,
        retriever=retriever,
        language="en",
        doc_id="test_doc",
        model=model,
        dataset_format="phenobert",
        include_details=True  # Include detailed output
    )

    # Print results
    print("\n=== MENTION EXTRACTION RESULTS ===")
    print(f"Result keys: {list(result.keys())}")
    print(f"Document ID: {result.get('doc_id', 'N/A')}")
    print(f"Language: {result.get('language', 'N/A')}")
    print(f"Dataset Format: {result.get('dataset_format', 'N/A')}")

    print(f"\nMentions found: {len(result['mentions'])}")
    for i, mention in enumerate(result['mentions'], 1):
        print(f"{i}. '{mention['text']}' (span: {mention['start_char']}-{mention['end_char']})")
        print(f"   Assertion: {mention['assertion']}")
        if mention['hpo_candidates']:
            top_candidate = mention['hpo_candidates'][0]
            print(f"   Top HPO: {top_candidate['hpo_id']} - {top_candidate['label']} (score: {top_candidate['score']:.3f})")

    print(f"\nMention Groups: {len(result['groups'])}")
    for i, group in enumerate(result['groups'], 1):
        print(f"Group {i}: {len(group['mentions'])} mentions")
        if group.get('final_hpo'):
            hpo = group['final_hpo']
            print(f"   Final HPO: {hpo['hpo_id']} - {hpo['label']} (score: {hpo['score']:.3f})")

    print(f"\nBenchmark Format: {len(result['benchmark_format'])} HPO terms")
    for hpo_id, assertion in result['benchmark_format']:
        print(f"  {hpo_id}: {assertion}")

    print("\n=== TEST COMPLETED SUCCESSFULLY ===")

if __name__ == "__main__":
    main()