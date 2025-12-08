from enum import Enum

import networkx as nx


class MatchType(Enum):
    """Types of hierarchical matches."""

    EXACT = "exact"
    ANCESTOR = "ancestor"  # Predicted is ancestor of gold
    DESCENDANT = "descendant"  # Predicted is descendant of gold
    SIBLING = "sibling"  # Share immediate parent
    COUSIN = "cousin"  # Share grandparent
    UNRELATED = "unrelated"  # No close relationship


class HierarchicalMatcher:
    """Match predictions with hierarchical relationships."""

    def __init__(self, hpo_graph: nx.DiGraph):
        self.hpo_graph = hpo_graph
        self.ic_scores = self._calculate_information_content()

    def classify_match(self, predicted_id: str, gold_id: str) -> MatchType:
        """Classify the relationship between predicted and gold HPO terms."""
        if predicted_id == gold_id:
            return MatchType.EXACT

        # Check ancestor/descendant relationships
        if self._is_ancestor(predicted_id, gold_id):
            return MatchType.ANCESTOR
        elif self._is_ancestor(gold_id, predicted_id):
            return MatchType.DESCENDANT

        # Check sibling/cousin relationships
        pred_parents = self._get_parents(predicted_id)
        gold_parents = self._get_parents(gold_id)

        if pred_parents & gold_parents:
            return MatchType.SIBLING

        pred_grandparents = self._get_grandparents(predicted_id)
        gold_grandparents = self._get_grandparents(gold_id)

        if pred_grandparents & gold_grandparents:
            return MatchType.COUSIN

        return MatchType.UNRELATED

    def calculate_partial_credit(
        self, match_type: MatchType, predicted_id: str, gold_id: str
    ) -> float:
        """
        Calculate partial credit based on match type and distance.
        Returns value between 0 and 1.
        """
        credit_map = {
            MatchType.EXACT: 1.0,
            MatchType.ANCESTOR: self._ancestor_credit(predicted_id, gold_id),
            MatchType.DESCENDANT: self._descendant_credit(predicted_id, gold_id),
            MatchType.SIBLING: 0.7,
            MatchType.COUSIN: 0.5,
            MatchType.UNRELATED: 0.0,
        }
        return credit_map[match_type]

    def _is_ancestor(self, ancestor_id: str, descendant_id: str) -> bool:
        """Check if ancestor_id is an ancestor of descendant_id."""
        try:
            return bool(nx.has_path(self.hpo_graph, ancestor_id, descendant_id))
        except nx.NetworkXError:
            return False

    def _get_parents(self, node_id: str) -> set:
        """Get immediate parents of a node."""
        try:
            return set(self.hpo_graph.predecessors(node_id))
        except nx.NetworkXError:
            return set()

    def _get_grandparents(self, node_id: str) -> set:
        """Get grandparents (parents of parents) of a node."""
        parents = self._get_parents(node_id)
        grandparents = set()
        for parent in parents:
            grandparents.update(self._get_parents(parent))
        return grandparents

    def _calculate_information_content(self) -> dict[str, float]:
        """Calculate information content scores for HPO terms."""
        # Placeholder: implement IC calculation
        return {}

    def _ancestor_credit(self, ancestor_id: str, descendant_id: str) -> float:
        """Credit based on specificity loss."""
        try:
            distance = nx.shortest_path_length(
                self.hpo_graph, ancestor_id, descendant_id
            )
            return float(max(0.5, 1.0 - (distance * 0.1)))
        except nx.NetworkXError:
            return 0.5

    def _descendant_credit(self, descendant_id: str, ancestor_id: str) -> float:
        """Credit based on over-specificity."""
        try:
            distance = nx.shortest_path_length(
                self.hpo_graph, ancestor_id, descendant_id
            )
            return float(max(0.7, 1.0 - (distance * 0.05)))
        except nx.NetworkXError:
            return 0.7
