from collections.abc import Hashable, Iterable

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from pgmpy import config
from pgmpy.base import DAG
from pgmpy.causal_discovery._base import BaseCausalDiscovery, _ScoreMixin
from pgmpy.structure_score import BaseStructureScore, get_scoring_method


class _Order:
    """
    Bookkeeping for a variable ordering and the DAG it implies during the GRaSP search.

    Ported from the ``Order`` helper of causal-learn's GRaSP implementation, adapted to work with
    variable names instead of column indices.
    """

    def __init__(self, variables: Iterable[Hashable]):
        self.order = list(variables)
        self.parents = {variable: [] for variable in self.order}
        self.local_scores = {variable: 0.0 for variable in self.order}
        self.n_edges = 0


class GRaSP(_ScoreMixin, BaseCausalDiscovery):
    """
    Score-based causal discovery using GRaSP (Greedy Relaxations of the Sparsest Permutation).

    GRaSP searches over variable orderings instead of over graph structures directly. For a given
    ordering, the DAG implied by it is obtained by greedily selecting, for each variable, the parent
    set among its predecessors that maximizes a decomposable structure score (e.g. BIC or BDeu);
    the score of the ordering is the sum of the local scores.

    Starting from a random ordering, the algorithm repeatedly applies **tuck** operations: a variable
    ``y`` is moved (together with the ancestors of ``y`` that lie between the two) immediately before
    one of its parents ``x``, reversing their relative order, after which the affected part of the
    ordering is re-scored. Tucks that strictly improve the total score are kept. Tucks that leave the
    score unchanged are explored recursively (up to ``depth`` levels) as free intermediate steps, where
    only tucks over *covered* edges (an edge ``x -> y`` is covered if the parents of ``x`` equal the
    parents of ``y`` minus ``x``) are allowed; at the top level any edge may be tucked. This is the
    greedy relaxation of the Sparsest Permutation algorithm, and the search terminates when no
    improving tuck can be found.

    This implementation is a port-with-attribution of ``GRaSP`` from causal-learn (MIT license): the
    ordering search, the depth-bounded tuck relaxation and the greedy grow-shrink parent selection are
    ported from ``causallearn/search/PermutationBased/GRaSP.py`` and ``gst.py``, adapted to pgmpy's
    ``BaseCausalDiscovery`` and ``StructureScore`` interfaces (pgmpy's own scoring is used; causal-learn
    is not vendored).

    Note: GRaSP belongs to the permutation-based family (SP/GSP/GRaSP/BOSS) which is under active
    development in pgmpy; its successor BOSS partially supersedes it. Upstream integration of this
    estimator should be coordinated with the maintainers of that family.

    Parameters
    ----------
    scoring_method : str or BaseStructureScore instance, default=None
        The score to be optimized during structure estimation. Please refer :doc:`/api/structure_score` for a list of
        available scoring methods.

        If ``None``, the appropriate scoring method is automatically selected based on the data type. If a string is
        provided, the corresponding scoring method is instantiated with default parameters. To customize score-specific
        parameters, please pass an instance of the scoring class.

    depth : int, default=3
        The maximum depth of the recursive search over chains of score-neutral tucks. Larger values loosen the
        relaxation (successively looser variants of GRaSP) at the cost of runtime. The reference implementation
        defaults to 3.

    max_iter : int or None, default=None
        Maximum number of accepted tucks. If None, the search runs until no score-improving tuck is found.

    return_type : str, default='dag'
        The type of graph to return. Options are:

        - 'dag': Returns a directed acyclic graph (DAG).
        - 'pdag': Returns a partially directed acyclic graph (PDAG).

    show_progress : bool, default=True
        If True, shows a progress bar while learning the causal structure.

    seed : int or None, default=None
        Seed for the random number generator used to draw the initial ordering and to shuffle the candidate tucks.

    Attributes
    ----------
    causal_graph_ : DAG or PDAG
        The learned causal graph as a directed acyclic graph (DAG) or partially directed acyclic graph (PDAG).

    adjacency_matrix_ : pd.DataFrame
        Adjacency matrix representation of the learned causal graph.

    scoring_method_ : BaseStructureScore
        The structure score used for learning, resolved from ``scoring_method``.

    causal_order_ : list
        The learned causal ordering; each variable's parents precede it in this list.

    n_features_in_ : int
        The number of features in the data used to learn the causal graph.

    feature_names_in_ : np.ndarray
        The feature names in the data used to learn the causal graph.

    Examples
    --------
    Simulate some data to use for causal discovery:

    >>> from pgmpy.example_models import load_model
    >>> model = load_model("bnlearn/cancer")
    >>> df = model.simulate(n_samples=1000, seed=42)

    Use the GRaSP algorithm to learn the causal structure from data:

    >>> from pgmpy.causal_discovery import GRaSP
    >>> est = GRaSP(scoring_method="bic-d")
    >>> est.fit(df)
    GRaSP(scoring_method='bic-d')
    >>> est.causal_graph_  # doctest: +ELLIPSIS
    <pgmpy.base.DAG.DAG object at 0x...>
    >>> est.n_features_in_
    5

    References
    ----------
    - Lam, W. Y., Andrews, B., & Ramsey, J. (2022). Greedy Relaxations of the Sparsest Permutation
      Algorithm. Proceedings of the 38th Conference on Uncertainty in Artificial Intelligence (UAI).
      `arXiv:2206.05421 <https://arxiv.org/abs/2206.05421>`__
    - `causal-learn <https://github.com/py-why/causal-learn>`__ (MIT license): this estimator ports the
      GRaSP search from ``causallearn/search/PermutationBased/GRaSP.py`` with attribution.
    """

    _tags = {
        "name": "grasp",
        "data_types": ["discrete", "continuous", "mixed"],
        "identifiable_graph": "cpdag",
        "requires_target": False,
        "capability:multivariate": True,
        "capability:expert_knowledge": [],
        "assumption:causal_sufficiency": True,
        "assumption:acyclicity": True,
        "assumption:faithfulness": False,
        "assumption:linearity": False,
        "assumption:additive_noise": False,
        "assumption:gaussian_noise": False,
        "assumption:non_gaussian_noise": False,
        "assumption:low_noise": False,
    }

    def __init__(
        self,
        scoring_method: str | BaseStructureScore | None = None,
        depth: int = 3,
        max_iter: int | None = None,
        return_type: str = "dag",
        show_progress: bool = True,
        seed: int | None = None,
    ):
        self.scoring_method = scoring_method
        self.depth = depth
        self.max_iter = max_iter
        self.return_type = return_type
        self.show_progress = show_progress
        self.seed = seed

    def _select_parents(
        self,
        y: Hashable,
        candidates: Iterable[Hashable],
    ) -> tuple[list[Hashable], float]:
        """
        Greedily select the parent set of `y` from `candidates`, maximizing the local score.

        Ported from the grow-shrink tree (GST) used by causal-learn's GRaSP: the grow phase repeatedly adds the
        candidate that most improves the local score, and the shrink phase then repeatedly removes the parent whose
        removal most improves it. Local scores are memoized by the structure score itself.

        Parameters
        ----------
        y : hashable
            The variable whose parent set is being selected.

        candidates : iterable of hashable
            The variables allowed as parents of `y` (its predecessors in the current ordering).

        Returns
        -------
        parents : list of hashable
            The selected parent set of `y`.

        local_score : float
            The local score of `y` given the selected parent set.
        """
        candidates = list(candidates)

        def score(parents: list[Hashable]) -> float:
            return self.scoring_method_.local_score(y, tuple(sorted(parents, key=repr)))

        parents = []
        local_score = score(parents)

        # Grow phase: keep adding the best improving candidate while any improves the score.
        while True:
            best_add, best_score = None, local_score
            for x in candidates:
                if x in parents:
                    continue
                x_score = score(parents + [x])
                if x_score > best_score:
                    best_add, best_score = x, x_score
            if best_add is None:
                break
            parents.append(best_add)
            local_score = best_score

        # Shrink phase: keep removing the best improving parent while any improves the score.
        while True:
            best_remove, best_score = None, local_score
            for x in parents:
                x_score = score([p for p in parents if p != x])
                if x_score > best_score:
                    best_remove, best_score = x, x_score
            if best_remove is None:
                break
            parents.remove(best_remove)
            local_score = best_score

        return parents, local_score

    @staticmethod
    def _get_ancestors(y: Hashable, ancestors: list[Hashable], order: _Order):
        """
        Append `y` and all of its ancestors (under the current parent sets) to `ancestors`.
        """
        ancestors.append(y)
        for x in order.parents[y]:
            if x not in ancestors:
                GRaSP._get_ancestors(x, ancestors, order)

    @staticmethod
    def _tuck(order: _Order, i: int, j: int):
        """
        Tuck the variable at position `i` (together with its ancestors between the two positions) into position `j`.
        """
        ancestors = []
        GRaSP._get_ancestors(order.order[i], ancestors, order)

        shift = 0
        for k in range(j + 1, i + 1):
            if order.order[k] in ancestors:
                order.order.insert(j + shift, order.order.pop(k))
                shift += 1

    def _rescore_window(self, order: _Order, j: int, i: int) -> tuple[int, float]:
        """
        Re-select the parent sets and local scores of the variables at positions `j` through `i`, which are the
        positions that a tuck of the variable at position `i` into position `j` can change.

        Returns
        -------
        edge_bump : int
            The change in the total number of edges.

        score_bump : float
            The change in the total score.
        """
        edge_bump = 0
        old_score = 0.0
        new_score = 0.0

        for k in range(j, i + 1):
            z = order.order[k]

            edge_bump -= len(order.parents[z])
            old_score += order.local_scores[z]

            z_parents, local_score = self._select_parents(z, order.order[:k])
            order.parents[z] = z_parents
            order.local_scores[z] = local_score

            edge_bump += len(z_parents)
            new_score += local_score

        return edge_bump, new_score - old_score

    def _find_improving_tuck(
        self,
        order: _Order,
        depth: int,
        flipped: set,
        history: list[set],
        rng: np.random.Generator,
    ) -> bool:
        """
        Depth-first search over tucks: return True as soon as a tuck (or a chain of score-neutral tucks followed by
        one tuck) that strictly improves the total score is found and applied, False if no such tuck exists.

        Parameters
        ----------
        order : _Order
            The current ordering, parent sets and local scores. Modified in place by accepted tucks; rejected tucks
            are rolled back.

        depth : int
            The remaining recursion depth. Deeper levels only consider tucks over covered edges.

        flipped : set
            The pairs of variables (as frozensets) whose relative order has been flipped by the chain of neutral tucks
            that led here.

        history : list of set
            The sets of flipped pairs already visited along the chain, used to avoid cycles.

        rng : numpy.random.Generator
            Source of randomness for visiting the candidate tucks.
        """
        cache: dict[int, tuple[Hashable, list[Hashable], float]] = {}
        cached_n_edges = 0

        for i in rng.permutation(len(order.order)):
            i = int(i)
            y = order.order[i]
            y_parents = [order.parents[y][k] for k in rng.permutation(len(order.parents[y]))]

            for x in y_parents:
                covered = {x, *order.parents[x]} == set(order.parents[y])

                # Only covered edges may be tucked at deeper levels of the relaxation.
                if len(history) > 0 and not covered:
                    continue

                j = order.order.index(x)

                for k in range(j, i + 1):
                    z = order.order[k]
                    cache[k] = (z, list(order.parents[z]), order.local_scores[z])
                cached_n_edges = order.n_edges

                self._tuck(order, i, j)
                edge_bump, score_bump = self._rescore_window(order, j, i)

                # Keep any tuck that strictly improves the total score.
                if score_bump > 1e-6:
                    order.n_edges += edge_bump
                    return True

                # A score-neutral tuck is a free intermediate step: follow it (up to `depth`) and look for a strictly
                # improving tuck from there.
                if score_bump > -1e-6:
                    flipped = flipped ^ {
                        frozenset((x, z)) for z in order.parents[x] if order.order.index(z) < i
                    }

                    if len(flipped) > 0 and flipped not in history:
                        history.append(flipped)
                        if depth > 0 and self._find_improving_tuck(order, depth - 1, flipped, history, rng):
                            return True
                        del history[-1]

                # Roll the tuck back before trying the next candidate.
                for k in range(j, i + 1):
                    z, z_parents, local_score = cache[k]
                    order.order[k] = z
                    order.parents[z] = z_parents
                    order.local_scores[z] = local_score
                order.n_edges = cached_n_edges

        return False

    def _fit(self, X: pd.DataFrame):
        """
        The fitting procedure for the GRaSP algorithm.

        Parameters
        ----------
        X : pandas.DataFrame
            The data to learn the causal structure from.

        Returns
        -------
        self : pgmpy.causal_discovery.GRaSP
            Returns the instance with the fitted attributes.
        """
        # Step 0: Check inputs
        if self.depth < 1:
            raise ValueError(f"depth must be at least 1, got {self.depth}.")
        if self.max_iter is not None and self.max_iter < 1:
            raise ValueError(f"max_iter must be at least 1 to accept at least one tuck, got {self.max_iter}.")
        if self.return_type.lower() not in ("dag", "pdag"):
            raise ValueError(f"return_type must be one of: dag, pdag. Got: {self.return_type}")

        # Step 1: Resolve the score and initialize the ordering with its implied parent sets.
        self.scoring_method_ = get_scoring_method(self.scoring_method, X)
        rng = np.random.default_rng(self.seed)

        order = _Order(list(self.feature_names_in_))
        order.order = [order.order[k] for k in rng.permutation(len(order.order))]

        for i, y in enumerate(order.order):
            order.parents[y], order.local_scores[y] = self._select_parents(y, order.order[:i])
            order.n_edges += len(order.parents[y])

        # Step 2: Repeatedly accept tucks that improve the total score until none can be found.
        n_tucks = 0
        pbar = tqdm(
            disable=not (self.show_progress and config.SHOW_PROGRESS),
            desc="Searching over orderings",
        )
        while self._find_improving_tuck(order, self.depth - 1, set(), [], rng):
            n_tucks += 1
            pbar.update(1)
            if self.max_iter is not None and n_tucks >= self.max_iter:
                break
        pbar.close()

        self.causal_order_ = list(order.order)

        # Step 3: Construct the DAG of the final ordering and assign attributes.
        current_model = DAG()
        current_model.add_nodes_from(order.order)
        current_model.add_edges_from(
            (x, y) for y in order.order for x in order.parents[y]
        )

        if self.return_type.lower() == "dag":
            self.causal_graph_ = current_model
        elif self.return_type.lower() == "pdag":
            self.causal_graph_ = current_model.to_pdag()

        self.adjacency_matrix_ = self.causal_graph_.to_adjacency(
            encoding="binary", nodelist=list(self.causal_graph_.nodes())
        )

        return self
