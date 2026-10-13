import numpy as np
import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

from pgmpy.base import DAG, PDAG
from pgmpy.causal_discovery import GRaSP
from pgmpy.example_models import load_model
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import DiscreteBayesianNetwork


def expected_failed_checks(estimator):
    return {
        "check_fit_score_takes_y": "Causal discovery estimators do not take y parameter in score method.",
        "check_n_features_in_after_fitting": "Failing for score method (not for fit) for unknown reason.",
    }


@parametrize_with_checks(
    [GRaSP(depth=2)],
    expected_failed_checks=expected_failed_checks,
)
def test_grasp_compatibility(estimator, check):
    check(estimator)


@pytest.fixture
def chain_model():
    model = DiscreteBayesianNetwork([("A", "B"), ("B", "C"), ("C", "D")])
    model.add_cpds(
        TabularCPD("A", 2, [[0.5], [0.5]]),
        TabularCPD("B", 2, [[0.05, 0.95], [0.95, 0.05]], evidence=["A"], evidence_card=[2]),
        TabularCPD("C", 2, [[0.05, 0.95], [0.95, 0.05]], evidence=["B"], evidence_card=[2]),
        TabularCPD("D", 2, [[0.05, 0.95], [0.95, 0.05]], evidence=["C"], evidence_card=[2]),
    )
    return model


@pytest.fixture
def chain_data(chain_model):
    return chain_model.simulate(n_samples=int(1e4), seed=7)


@pytest.fixture
def cancer_model():
    return load_model("bnlearn/cancer")


@pytest.fixture
def cancer_data(cancer_model):
    return cancer_model.simulate(n_samples=1000, seed=42)


def pdag_adjacencies(pdag):
    """Adjacency pairs of a PDAG's edges, ignoring the direction in which undirected edges are stored."""
    return {frozenset((u, v)) for u, v, _ in pdag.get_edges(data=True)}


class TestGRaSP:
    def test_chain_recovery_and_attributes(self, chain_data, chain_model):
        # The chain A -> B -> C -> D has no unshielded collider, so its Markov equivalence
        # class contains every collider-free orientation of the skeleton: the learned DAG
        # must be a member of the class, i.e. map to the same (fully undirected) CPDAG.
        est = GRaSP(scoring_method="bdeu", return_type="pdag", seed=42)
        est.fit(chain_data)

        assert isinstance(est.causal_graph_, PDAG)
        assert pdag_adjacencies(est.causal_graph_) == pdag_adjacencies(DAG(chain_model.edges()).to_pdag())
        assert sorted(est.causal_order_) == sorted(chain_data.columns)

        assert est.n_features_in_ == chain_data.shape[1]
        assert list(est.feature_names_in_) == list(chain_data.columns)

        adj = est.adjacency_matrix_
        assert sorted(adj.index) == sorted(chain_data.columns)
        assert sorted(adj.columns) == sorted(chain_data.columns)
        assert np.all(np.diag(adj) == 0)

    def test_cancer_skeleton_recovery(self, cancer_data, cancer_model):
        # The cancer network contains a weak edge whose orientation BDeu does not identify at
        # n=1000, so only the skeleton (and the estimator attributes) are asserted here.
        est = GRaSP(scoring_method="bdeu", return_type="dag", seed=42)
        est.fit(cancer_data)

        assert isinstance(est.causal_graph_, DAG)
        assert {frozenset(edge) for edge in est.causal_graph_.edges()} == {
            frozenset(edge) for edge in cancer_model.edges()
        }
        assert est.n_features_in_ == cancer_data.shape[1]
        assert list(est.feature_names_in_) == list(cancer_data.columns)

    def test_returns_pdag(self, cancer_data):
        est = GRaSP(scoring_method="bdeu", return_type="pdag")
        est.fit(cancer_data)
        assert isinstance(est.causal_graph_, PDAG)

    def test_seed_and_max_iter(self, cancer_data):
        est1 = GRaSP(scoring_method="bdeu", max_iter=5, seed=42)
        est2 = GRaSP(scoring_method="bdeu", max_iter=5, seed=42)
        est1.fit(cancer_data)
        est2.fit(cancer_data)
        assert set(est1.causal_graph_.edges()) == set(est2.causal_graph_.edges())
        assert est1.causal_order_ == est2.causal_order_
