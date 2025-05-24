# Ontology Similarity

Phentrieve includes functionality to calculate semantic similarity between HPO terms using the ontology graph structure. This is useful for understanding relationships between different phenotypes and evaluating the precision of term mapping.

## Semantic Similarity in Ontologies

Semantic similarity in ontologies measures how close two terms are in meaning, based on their positions in the ontology graph and their information content. Several established methods exist for calculating this similarity:

### Information Content (IC)

Information Content (IC) is a measure of how specific a term is within the ontology. The more specific a term, the higher its IC value.

IC is calculated based on the frequency of the term in a reference corpus or based on the structure of the ontology. In the structural approach, terms with many descendants (more general terms) have lower IC values than terms with few descendants (more specific terms).

### Similarity Metrics

Phentrieve supports several similarity metrics:

#### Resnik Similarity

Measures similarity based on the Information Content (IC) of the Most Informative Common Ancestor (MICA) of two terms:

```math
sim_resnik(t1, t2) = IC(MICA(t1, t2))
```

#### Lin Similarity

Normalizes Resnik similarity by the information content of the two terms:

```math
sim_lin(t1, t2) = 2 * IC(MICA(t1, t2)) / (IC(t1) + IC(t2))
```

#### Jiang-Conrath Similarity

Considers both the information content of the terms and their MICA:

```math
sim_jc(t1, t2) = 1 / (1 + IC(t1) + IC(t2) - 2 * IC(MICA(t1, t2)))
```

#### Hybrid Similarity (Default)

Phentrieve's default hybrid approach combines elements of multiple metrics for more balanced results.

## Using Ontology Similarity in Phentrieve

### CLI Usage

Calculate semantic similarity between two specific HPO terms:

```bash
# Calculate similarity between two HPO terms using the default 'hybrid' formula
phentrieve similarity calculate HP:0001250 HP:0001251

# Specify a different similarity formula
phentrieve similarity calculate HP:0001250 HP:0001251 --formula resnik
```

Available formulas:

- `hybrid` (default): Phentrieve's custom hybrid approach
- `resnik`: Resnik similarity
- `lin`: Lin similarity
- `jc`: Jiang-Conrath similarity
- `ic`: Information Content only

### Programmatic Usage

```python
from phentrieve.similarity import calculate_similarity

# Calculate similarity between two HPO terms
similarity = calculate_similarity("HP:0001250", "HP:0001251", formula="hybrid")
print(f"Similarity: {similarity}")
```

## Applications of Ontology Similarity

Ontology similarity in Phentrieve serves several purposes:

1. **Result Evaluation**: Assess how close a retrieved HPO term is to the ground truth term
2. **Term Grouping**: Group similar HPO terms for more concise reporting
3. **Query Expansion**: Find related terms to expand search queries
4. **Hierarchical Navigation**: Navigate the HPO hierarchy to find more general or specific terms

!!! note "Precomputed Graph Data"
    Phentrieve precomputes and caches graph data (HPO term depths, ancestor counts, etc.) to make similarity calculations more efficient.
