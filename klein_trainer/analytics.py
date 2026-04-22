"""
Training analytics module for Klein LoRA Trainer.
Provides persistent logging, per-sample tracking, and trend analysis.
"""

import json
import threading
import os
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field, asdict
from collections import deque


@dataclass
class PerSampleLoss:
    """Per-sample loss data point."""
    step: int
    path: str
    loss: float
    timestep: float  # Flow matching timestep [0, 1]


@dataclass
class StepMetrics:
    """Metrics for a single training step."""
    step: int
    loss: float
    lr: float
    ema_10: Optional[float] = None
    ema_50: Optional[float] = None
    ema_100: Optional[float] = None
    trend_slope: Optional[float] = None
    trend_status: str = "unknown"  # "improving", "stable", "degrading", "converged"


class EMACalculator:
    """Exponential Moving Average calculator with configurable window."""

    def __init__(self, window: int):
        self.window = window
        self.alpha = 2.0 / (window + 1)
        self.value: Optional[float] = None

    def update(self, new_value: float) -> float:
        """Update EMA with new value and return current EMA."""
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.alpha * new_value + (1 - self.alpha) * self.value
        return self.value

    def reset(self):
        """Reset EMA state."""
        self.value = None

    def get_value(self) -> Optional[float]:
        """Get current EMA value."""
        return self.value


class LinearRegressionWindow:
    """Sliding window linear regression for trend detection."""

    def __init__(self, window_size: int = 200):
        self.window_size = window_size
        self.steps: deque = deque(maxlen=window_size)
        self.losses: deque = deque(maxlen=window_size)

    def add(self, step: int, loss: float):
        """Add a data point to the window."""
        self.steps.append(step)
        self.losses.append(loss)

    def get_slope(self) -> Optional[float]:
        """Calculate slope using least squares regression."""
        n = len(self.steps)
        if n < 10:  # Need minimum data points
            return None

        # Convert to lists for calculation
        x = list(self.steps)
        y = list(self.losses)

        # Calculate means
        x_mean = sum(x) / n
        y_mean = sum(y) / n

        # Calculate slope: sum((x-x_mean)(y-y_mean)) / sum((x-x_mean)^2)
        numerator = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
        denominator = sum((xi - x_mean) ** 2 for xi in x)

        if denominator == 0:
            return 0.0

        return numerator / denominator

    def get_r_squared(self) -> Optional[float]:
        """Calculate R-squared (coefficient of determination)."""
        n = len(self.steps)
        if n < 10:
            return None

        slope = self.get_slope()
        if slope is None:
            return None

        x = list(self.steps)
        y = list(self.losses)

        x_mean = sum(x) / n
        y_mean = sum(y) / n

        # Predicted values
        y_pred = [y_mean + slope * (xi - x_mean) for xi in x]

        # SS_res and SS_tot
        ss_res = sum((yi - ypi) ** 2 for yi, ypi in zip(y, y_pred))
        ss_tot = sum((yi - y_mean) ** 2 for yi in y)

        if ss_tot == 0:
            return 1.0

        return 1 - (ss_res / ss_tot)

    def reset(self):
        """Clear the window."""
        self.steps.clear()
        self.losses.clear()


class OutlierDetector:
    """IQR-based outlier detection for problematic images."""

    def __init__(self, iqr_multiplier: float = 1.5):
        self.iqr_multiplier = iqr_multiplier
        self.per_image_losses: Dict[str, List[float]] = {}

    def add_sample(self, path: str, loss: float):
        """Record a loss value for an image."""
        if path not in self.per_image_losses:
            self.per_image_losses[path] = []
        self.per_image_losses[path].append(loss)

    def get_image_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for each image."""
        stats = {}
        for path, losses in self.per_image_losses.items():
            if losses:
                stats[path] = {
                    'mean': sum(losses) / len(losses),
                    'min': min(losses),
                    'max': max(losses),
                    'count': len(losses),
                    'variance': self._variance(losses) if len(losses) > 1 else 0.0
                }
        return stats

    def _variance(self, values: List[float]) -> float:
        """Calculate variance of a list of values."""
        n = len(values)
        if n < 2:
            return 0.0
        mean = sum(values) / n
        return sum((x - mean) ** 2 for x in values) / (n - 1)

    def get_outliers(self, top_n: int = 10) -> List[Tuple[str, float, float]]:
        """
        Returns list of (path, mean_loss, z_score) for outlier images.
        Uses IQR method to detect outliers.
        """
        if not self.per_image_losses:
            return []

        # Calculate mean loss per image
        mean_losses = {
            path: sum(losses) / len(losses)
            for path, losses in self.per_image_losses.items()
            if losses
        }

        if len(mean_losses) < 4:
            # Not enough data for IQR, return top by raw mean
            sorted_items = sorted(mean_losses.items(), key=lambda x: x[1], reverse=True)
            return [(path, loss, 0.0) for path, loss in sorted_items[:top_n]]

        # Calculate quartiles
        sorted_losses = sorted(mean_losses.values())
        n = len(sorted_losses)
        q1_idx = n // 4
        q3_idx = 3 * n // 4
        q1 = sorted_losses[q1_idx]
        q3 = sorted_losses[q3_idx]
        iqr = q3 - q1

        upper_bound = q3 + self.iqr_multiplier * iqr

        # Find outliers and non-outliers above median
        outliers = []
        median = sorted_losses[n // 2]

        for path, mean_loss in mean_losses.items():
            if iqr > 0:
                # Calculate how many IQRs above Q3
                z_score = (mean_loss - q3) / iqr if mean_loss > q3 else 0.0
            else:
                z_score = 0.0

            if mean_loss > upper_bound:
                outliers.append((path, mean_loss, z_score))
            elif mean_loss > median:
                # Include high-but-not-outlier samples with z_score=0
                outliers.append((path, mean_loss, 0.0))

        # Sort by mean loss descending and return top_n
        outliers.sort(key=lambda x: x[1], reverse=True)
        return outliers[:top_n]

    def reset(self):
        """Clear all recorded data."""
        self.per_image_losses.clear()


class TrainingAnalytics:
    """
    Main analytics class for Klein LoRA training.

    Features:
    - Persistent JSONL logging with auto-resume
    - Per-sample loss tracking with image path association
    - EMA calculation with configurable windows (10, 50, 100)
    - Linear regression trend detection
    - Convergence detection
    - Outlier detection for problematic images
    - Thread-safe operations
    """

    def __init__(
        self,
        output_dir: str,
        output_name: str,
        ema_windows: Tuple[int, ...] = (10, 50, 100),
        trend_window: int = 200,
        convergence_threshold: float = 0.0001,
        convergence_steps: int = 100,
        enable_per_sample: bool = True,
    ):
        """
        Initialize training analytics.

        Args:
            output_dir: Base output directory
            output_name: Name of this training run
            ema_windows: Tuple of EMA window sizes
            trend_window: Window size for linear regression
            convergence_threshold: Slope threshold for convergence detection
            convergence_steps: Number of steps to check for convergence
            enable_per_sample: Whether to track per-sample losses
        """
        # Paths
        self.output_dir = Path(output_dir) / output_name
        self.metrics_file = self.output_dir / "metrics.jsonl"
        self.per_sample_file = self.output_dir / "per_sample_losses.jsonl"

        # Settings
        self.enable_per_sample = enable_per_sample
        self.ema_windows = ema_windows

        # Create directory if needed
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Thread safety
        self._lock = threading.RLock()

        # EMA calculators
        self.ema_calculators = {w: EMACalculator(w) for w in ema_windows}

        # Trend analysis
        self.trend_analyzer = LinearRegressionWindow(trend_window)
        self.convergence_threshold = convergence_threshold
        self.convergence_steps = convergence_steps

        # Outlier detection
        self.outlier_detector = OutlierDetector()

        # In-memory history for UI
        self.metrics_history: List[StepMetrics] = []

        # Convergence tracking
        self._recent_slopes: deque = deque(maxlen=convergence_steps)

        # Resume state
        self.last_step = 0
        self._load_existing_metrics()

    def _load_existing_metrics(self):
        """Load existing metrics for resume support."""
        if not self.metrics_file.exists():
            return

        with self._lock:
            try:
                with open(self.metrics_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            metrics = StepMetrics(
                                step=data['step'],
                                loss=data['loss'],
                                lr=data['lr'],
                                ema_10=data.get('ema_10'),
                                ema_50=data.get('ema_50'),
                                ema_100=data.get('ema_100'),
                                trend_slope=data.get('trend_slope'),
                                trend_status=data.get('trend_status', 'unknown'),
                            )
                            self.metrics_history.append(metrics)

                            # Update EMA calculators to restore state
                            for window, calc in self.ema_calculators.items():
                                calc.update(data['loss'])

                            # Update trend analyzer
                            self.trend_analyzer.add(data['step'], data['loss'])

                            self.last_step = max(self.last_step, data['step'])
                        except (json.JSONDecodeError, KeyError) as e:
                            print(f"Warning: Skipping malformed metrics line: {e}")
                            continue

                print(f"Analytics: Loaded {len(self.metrics_history)} steps from previous run (last step: {self.last_step})")
            except Exception as e:
                print(f"Warning: Error loading metrics file: {e}")

    def record_step(
        self,
        step: int,
        loss: float,
        lr: float,
        per_sample_losses: Optional[List[Tuple[str, float, float]]] = None,
    ) -> StepMetrics:
        """
        Record metrics for a training step.

        Args:
            step: Current training step
            loss: Mean loss for this step
            lr: Current learning rate
            per_sample_losses: Optional list of (path, loss, timestep) tuples

        Returns:
            StepMetrics with computed analytics
        """
        with self._lock:
            # Fresh run detection: if step 1 is recorded but we have old data,
            # this is a new training run reusing the same output name.
            # Clear EVERYTHING so the new run starts clean.
            if step == 1 and self.last_step > 0:
                print("Analytics: New training run detected, clearing stale metrics")
                self.outlier_detector.reset()
                self.metrics_history.clear()
                self._recent_slopes.clear()
                # Reset EMA calculators
                self.ema_calculators = {w: EMACalculator(w) for w in self.ema_windows}
                # Reset trend analyzer
                self.trend_analyzer = LinearRegressionWindow(self.trend_analyzer.window_size)
                # Truncate metric files
                if self.per_sample_file.exists():
                    self.per_sample_file.unlink()
                if self.metrics_file.exists():
                    self.metrics_file.unlink()
                self.last_step = 0

            # Skip if already recorded (resume case)
            if step <= self.last_step:
                if self.metrics_history:
                    return self.metrics_history[-1]
                return StepMetrics(step=step, loss=loss, lr=lr)

            # Calculate EMAs
            ema_values = {}
            for window, calc in self.ema_calculators.items():
                ema_values[window] = calc.update(loss)

            # Update trend analyzer
            self.trend_analyzer.add(step, loss)
            slope = self.trend_analyzer.get_slope()

            # Determine trend status
            trend_status = self._determine_trend_status(slope)

            # Create metrics object
            metrics = StepMetrics(
                step=step,
                loss=loss,
                lr=lr,
                ema_10=ema_values.get(10),
                ema_50=ema_values.get(50),
                ema_100=ema_values.get(100),
                trend_slope=slope,
                trend_status=trend_status,
            )

            # Store in memory
            self.metrics_history.append(metrics)
            self.last_step = step

            # Persist to file
            self._write_metrics_line(metrics)

            # Record per-sample losses if provided and enabled
            if self.enable_per_sample and per_sample_losses:
                for path, sample_loss, timestep in per_sample_losses:
                    self.outlier_detector.add_sample(path, sample_loss)
                    self._write_per_sample_line(PerSampleLoss(
                        step=step,
                        path=path,
                        loss=sample_loss,
                        timestep=timestep,
                    ))

            return metrics

    def _determine_trend_status(self, slope: Optional[float]) -> str:
        """Determine trend status from slope."""
        if slope is None:
            return "unknown"

        self._recent_slopes.append(slope)

        # Check for convergence (slope near zero for extended period)
        if len(self._recent_slopes) >= self.convergence_steps:
            avg_slope = sum(self._recent_slopes) / len(self._recent_slopes)
            if abs(avg_slope) < self.convergence_threshold:
                return "converged"

        # Determine direction based on slope magnitude
        # Use a threshold relative to typical loss values
        if slope < -self.convergence_threshold:
            return "improving"
        elif slope > self.convergence_threshold:
            return "degrading"
        else:
            return "stable"

    def _write_metrics_line(self, metrics: StepMetrics):
        """Append metrics to JSONL file."""
        try:
            with open(self.metrics_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(asdict(metrics)) + '\n')
        except Exception as e:
            print(f"Warning: Failed to write metrics: {e}")

    def _write_per_sample_line(self, sample_loss: PerSampleLoss):
        """Append per-sample loss to JSONL file."""
        if not self.enable_per_sample:
            return
        try:
            with open(self.per_sample_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(asdict(sample_loss)) + '\n')
        except Exception as e:
            print(f"Warning: Failed to write per-sample loss: {e}")

    def get_loss_history(self) -> List[Tuple[int, float]]:
        """Get loss history as list of (step, loss) tuples."""
        with self._lock:
            return [(m.step, m.loss) for m in self.metrics_history]

    def get_outlier_images(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """
        Get top N problematic images based on loss.

        Returns:
            List of dicts with path, mean_loss, z_score, filename
        """
        with self._lock:
            outliers = self.outlier_detector.get_outliers(top_n)
            return [
                {
                    "path": path,
                    "filename": Path(path).name,
                    "mean_loss": round(mean_loss, 4),
                    "z_score": round(z_score, 2),
                }
                for path, mean_loss, z_score in outliers
            ]

    def get_ema_series(self, window: int) -> List[Tuple[int, float]]:
        """Get EMA series for a specific window."""
        with self._lock:
            key = f"ema_{window}"
            return [
                (m.step, getattr(m, key))
                for m in self.metrics_history
                if getattr(m, key, None) is not None
            ]

    def get_current_trend(self) -> Dict[str, Any]:
        """Get current trend analysis."""
        with self._lock:
            if not self.metrics_history:
                return {"status": "unknown", "slope": None}

            latest = self.metrics_history[-1]
            return {
                "status": latest.trend_status,
                "slope": latest.trend_slope,
                "r_squared": self.trend_analyzer.get_r_squared(),
                "ema_10": latest.ema_10,
                "ema_50": latest.ema_50,
                "ema_100": latest.ema_100,
            }

    def get_latest_metrics(self) -> Optional[StepMetrics]:
        """Get the most recent metrics."""
        with self._lock:
            return self.metrics_history[-1] if self.metrics_history else None

    def export_metrics(self, format: str = "json") -> str:
        """Export all metrics to a format for external tools."""
        with self._lock:
            if format == "json":
                return json.dumps([asdict(m) for m in self.metrics_history], indent=2)
            elif format == "csv":
                if not self.metrics_history:
                    return ""
                headers = list(asdict(self.metrics_history[0]).keys())
                lines = [",".join(headers)]
                for m in self.metrics_history:
                    values = []
                    for v in asdict(m).values():
                        if v is None:
                            values.append("")
                        else:
                            values.append(str(v))
                    lines.append(",".join(values))
                return "\n".join(lines)
            else:
                raise ValueError(f"Unknown export format: {format}")

    def reset(self):
        """Reset all analytics state (for new training run)."""
        with self._lock:
            self.metrics_history.clear()
            self._recent_slopes.clear()
            self.last_step = 0

            for calc in self.ema_calculators.values():
                calc.reset()

            self.trend_analyzer.reset()
            self.outlier_detector.reset()

            # Clear files
            if self.metrics_file.exists():
                self.metrics_file.unlink()
            if self.per_sample_file.exists():
                self.per_sample_file.unlink()
