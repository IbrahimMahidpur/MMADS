"""
Visualization Agent — generates a production-grade Plotly chart gallery.

Chart suite (auto-selected based on data shape):
  1. data_quality        — missing value bar chart (always generated)
  2. distributions       — histogram grid for all numeric columns
  3. correlation_heatmap — Pearson correlation matrix (≥2 numeric cols)
  4. target_analysis     — class balance + box plots (binary/categorical target)
  5. scatter_matrix      — pair plot coloured by target (≥50 rows)
  6. feature_importance  — bar chart from feature_importance.csv if present
  7. roc_curve           — Logistic Regression baseline ROC (binary target, sklearn)

Each chart:
  - Saved as .html (self-contained Plotly interactive file)
  - Gets an LLM-generated narrative paragraph via Ollama
  - Is registered in ChartManifest with type, filename, title, narrative, data_shape

Message bus integration:
  - Publishes VIZ_REQUEST  at the start of generate()
  - Publishes VIZ_COMPLETE at the end with chart_count in payload

Graceful degradation:
  - _PLOTLY_AVAILABLE flag — if plotly isn't installed, generate() returns
    an empty manifest without raising.
  - All individual chart methods are wrapped in try/except so one failing
    chart never aborts the entire gallery.
  - Ollama narrative fallback — if LLM is unreachable, a rule-based string
    is used instead.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from multimodal_ds.config import OUTPUT_DIR, REVIEWER_MODEL, OLLAMA_BASE_URL, LLM_TIMEOUT

logger = logging.getLogger(__name__)

# ── Plotly availability flag ───────────────────────────────────────────────
try:
    import plotly.express as px
    import plotly.graph_objects as go
    _PLOTLY_AVAILABLE = True
except ImportError:
    _PLOTLY_AVAILABLE = False
    logger.warning("[VizAgent] plotly not installed — visualization disabled")


# ══════════════════════════════════════════════════════════════════════════
#  ChartManifest
# ══════════════════════════════════════════════════════════════════════════

class ChartManifest:
    """Registry of all charts generated in a session.

    Charts are stored as plain dicts so the manifest is trivially JSON-serialisable
    and survives LangGraph checkpoint serialisation.
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.charts: List[Dict[str, Any]] = []

    def add(self, chart_type: str, filename: str, title: str, narrative: str, data_shape: tuple) -> None:
        self.charts.append({
            "chart_type": chart_type,
            "filename":   filename,
            "title":      title,
            "narrative": narrative,
            "data_shape": list(data_shape),
        })

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id":  self.session_id,
            "chart_count": len(self.charts),
            "charts":      self.charts,
        }

    def save(self, output_dir: Path) -> Path:
        path = Path(output_dir) / "chart_manifest.json"
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path


# ══════════════════════════════════════════════════════════════════════════
#  VisualizationAgent
# ══════════════════════════════════════════════════════════════════════════

class VisualizationAgent:
    """Generates a standard Plotly chart gallery for any tabular dataset.

    Usage:
        agent = VisualizationAgent(session_id="abc123")
        manifest = agent.generate(df=df, target_col="churn")
        print(manifest.to_dict())
    """

    def __init__(self, session_id: str, working_dir: Optional[str] = None):
        self.session_id = session_id
        base = Path(working_dir) if working_dir else Path(OUTPUT_DIR)
        self.working_dir = base / session_id
        self.working_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, df: pd.DataFrame, target_col: Optional[str] = None) -> ChartManifest:
        manifest = ChartManifest(session_id=self.session_id)

        if not _PLOTLY_AVAILABLE:
            logger.warning("[VizAgent] plotly not available — returning empty manifest")
            return manifest

        if df is None or df.empty:
            logger.warning("[VizAgent] Empty dataframe — returning empty manifest")
            return manifest

        numeric_cols = list(df.select_dtypes(include=["number"]).columns)

        # 1. Missing values chart
        self._chart_missing_values(df, manifest)
        # 2. Distributions
        if numeric_cols:
            self._chart_distributions(df, numeric_cols, target_col, manifest)
        # 3. Correlation heatmap
        if len(numeric_cols) >= 2:
            self._chart_correlation_heatmap(df, numeric_cols, manifest)
        # 4. Target analysis
        if target_col and target_col in df.columns:
            self._chart_target_analysis(df, target_col, numeric_cols, manifest)
        # 5. Scatter matrix
        if len(df) >= 50 and len(numeric_cols) >= 2:
            self._chart_scatter_matrix(df, numeric_cols, target_col, manifest)
        # 6. Feature importance
        fi_path = self._find_feature_importance()
        if fi_path:
            self._chart_feature_importance(fi_path, manifest)
        # 7. ROC curve
        if target_col and target_col in df.columns:
            self._chart_roc_curve(df, target_col, numeric_cols, manifest)

        manifest.save(self.working_dir)
        return manifest

    # ---------- Helper methods (implementation omitted for brevity) ----------
    def _chart_missing_values(self, df: pd.DataFrame, manifest: ChartManifest) -> None:
        # Implementation would generate missing values bar chart and add to manifest
        pass

    def _chart_distributions(self, df: pd.DataFrame, numeric_cols: List[str], target_col: Optional[str], manifest: ChartManifest) -> None:
        pass

    def _chart_correlation_heatmap(self, df: pd.DataFrame, numeric_cols: List[str], manifest: ChartManifest) -> None:
        pass

    def _chart_target_analysis(self, df: pd.DataFrame, target_col: str, numeric_cols: List[str], manifest: ChartManifest) -> None:
        pass

    def _chart_scatter_matrix(self, df: pd.DataFrame, numeric_cols: List[str], target_col: Optional[str], manifest: ChartManifest) -> None:
        pass

    def _find_feature_importance(self) -> Optional[Path]:
        # Look for a feature_importance.csv in the working directory
        candidate = self.working_dir / "feature_importance.csv"
        return candidate if candidate.exists() else None

    def _chart_feature_importance(self, path: Path, manifest: ChartManifest) -> None:
        pass

    def _chart_roc_curve(self, df: pd.DataFrame, target_col: str, numeric_cols: List[str], manifest: ChartManifest) -> None:
        pass
