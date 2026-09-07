"""Write-ahead admissions for bounded RNA experiments in the workflow journal."""

import hashlib
from typing import Any

from .. import record_io
from . import journal
from .models import AutomatedWorkflowConfig


class CandidateBudgetExceeded(ValueError):
    """The next scientific experiment exceeds an explicit execution limit."""


def candidate_identity(inputs: dict[str, Any], *, graph: bool = False) -> str:
    """Identify exact numerical inputs, omitting resolution for graph reuse."""
    values = dict(inputs)
    parameters = dict(values["parameters"])
    parameters.pop("candidateId", None)
    if graph:
        parameters.pop("leidenResolution", None)
    values["parameters"] = parameters
    return hashlib.sha256(record_io.canonical_json_bytes(values)).hexdigest()


class CandidateBudget:
    """Reconstruct bounded admissions from fixed immutable journal slots."""

    def __init__(
        self,
        store: Any,
        prefix: str,
        workflow_run_id: str,
        config: AutomatedWorkflowConfig,
        provenance: dict[str, Any],
    ) -> None:
        self.store = store
        self.prefix = prefix
        self.workflow_run_id = workflow_run_id
        self.config = config
        self.provenance = provenance
        self.admissions: dict[str, list[dict[str, Any]]] = {}
        for scope in ("sample0", "sample1", "full"):
            limit = (
                config.maxFullPartitions
                if scope == "full"
                else config.maxScreeningEvaluations
            )
            rows: list[dict[str, Any]] = []
            for slot in range(limit):
                row = journal.load_checkpoint(
                    store,
                    prefix,
                    workflow_run_id,
                    self._key(scope, slot, "admission"),
                    inputs=provenance,
                )
                if row is None:
                    continue
                if (
                    row.get("slot") != slot
                    or row.get("scope") != scope
                    or slot != len(rows)
                ):
                    raise ValueError("Candidate admission history is inconsistent")
                rows.append(row)
            self.admissions[scope] = rows

    @staticmethod
    def _key(scope: str, slot: int, kind: str) -> str:
        return f"parameter_tuning/{scope}/evaluation{slot}/{kind}"

    def admit(self, scope: str, inputs: dict[str, Any]) -> dict[str, Any]:
        if scope not in self.admissions:
            raise ValueError("Unknown candidate execution scope")
        identity = candidate_identity(inputs)
        rows = self.admissions[scope]
        for row in rows:
            if row["identity"] == identity:
                return row
        graph_identity = candidate_identity(inputs, graph=True)
        if scope == "full":
            if len(rows) >= self.config.maxFullPartitions:
                raise CandidateBudgetExceeded("Full-cohort partition limit reached")
            graphs = {row["graphIdentity"] for row in rows} | {graph_identity}
            if len(graphs) > self.config.maxFullGraphs:
                raise CandidateBudgetExceeded("Full-cohort graph limit reached")
        else:
            if len(rows) >= self.config.maxScreeningEvaluations:
                raise CandidateBudgetExceeded("Screening candidate limit reached")
            total = sum(len(self.admissions[name]) for name in ("sample0", "sample1"))
            if total >= self.config.maxTotalScreeningEvaluations:
                raise CandidateBudgetExceeded("Total screening candidate limit reached")
        row = {
            "slot": len(rows),
            "scope": scope,
            "identity": identity,
            "graphIdentity": graph_identity,
            "executionInputs": inputs,
        }
        journal.save_checkpoint(
            self.store,
            self.prefix,
            self.workflow_run_id,
            self._key(scope, len(rows), "admission"),
            inputs=self.provenance,
            outputs=row,
        )
        rows.append(row)
        return row

    def check_many(
        self, scope: str, inputs: list[dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        """Check room for a comparison without committing an admission."""
        existing = self.admissions[scope]
        identities = {row["identity"] for row in existing}
        pending = {
            candidate_identity(value): value
            for value in inputs
            if candidate_identity(value) not in identities
        }
        total = len(existing) + len(pending)
        if scope == "full":
            graphs = {row["graphIdentity"] for row in existing} | {
                candidate_identity(value, graph=True) for value in pending.values()
            }
            if (
                total > self.config.maxFullPartitions
                or len(graphs) > self.config.maxFullGraphs
            ):
                raise CandidateBudgetExceeded(
                    "The full-cohort comparison exceeds the remaining graph/partition limit"
                )
        elif (
            total > self.config.maxScreeningEvaluations
            or sum(len(self.admissions[name]) for name in ("sample0", "sample1"))
            + len(pending)
            > self.config.maxTotalScreeningEvaluations
        ):
            raise CandidateBudgetExceeded(
                "The screening comparison exceeds the remaining candidate limit"
            )
        return pending

    def admit_many(self, scope: str, inputs: list[dict[str, Any]]) -> None:
        """Check a baseline or matched pair in full before its first computation."""
        for value in self.check_many(scope, inputs).values():
            self.admit(scope, value)

    def completed(self, admission: dict[str, Any]) -> dict[str, Any] | None:
        return journal.load_checkpoint(
            self.store,
            self.prefix,
            self.workflow_run_id,
            self._key(admission["scope"], admission["slot"], "complete"),
            inputs=admission,
        )

    def complete(self, admission: dict[str, Any], output: dict[str, Any]) -> None:
        journal.save_checkpoint(
            self.store,
            self.prefix,
            self.workflow_run_id,
            self._key(admission["scope"], admission["slot"], "complete"),
            inputs=admission,
            outputs=output,
        )

    def summary(self) -> dict[str, Any]:
        """Count distinct reserved and completed comparisons, including reuse."""
        scopes = {}
        for scope, rows in self.admissions.items():
            completed = [row for row in rows if self.completed(row) is not None]
            scopes[scope] = {
                state: {
                    "graphs": len({row["graphIdentity"] for row in entries}),
                    "partitions": len({row["identity"] for row in entries}),
                }
                for state, entries in (("reserved", rows), ("completed", completed))
            }
        return {
            "scopes": scopes,
            "limits": {
                "perScreen": self.config.maxScreeningEvaluations,
                "totalScreens": self.config.maxTotalScreeningEvaluations,
                "fullPartitions": self.config.maxFullPartitions,
                "fullGraphs": self.config.maxFullGraphs,
            },
        }
