"""
Claude API integration for image captioning and training analysis.
"""

import base64
import json
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


class ClaudeAPI:
    """
    Wrapper for Claude API with vision capabilities.

    Provides:
    - Connection testing
    - Image captioning/description
    - Training run analysis
    """

    # Default model for vision tasks
    DEFAULT_MODEL = "claude-sonnet-4-20250514"

    def __init__(self, api_key: str):
        """
        Initialize Claude API client.

        Args:
            api_key: Anthropic API key
        """
        self.api_key = api_key
        self.client = None

        if not ANTHROPIC_AVAILABLE:
            raise ImportError(
                "anthropic package not installed. "
                "Run: pip install anthropic"
            )

        if api_key:
            self.client = anthropic.Anthropic(api_key=api_key)

    def test_connection(self) -> Tuple[bool, str]:
        """
        Test the API connection with a simple request.

        Returns:
            Tuple of (success: bool, message: str)
        """
        if not self.api_key:
            return False, "No API key provided"

        if not self.client:
            return False, "Client not initialized"

        try:
            # Simple test message
            response = self.client.messages.create(
                model=self.DEFAULT_MODEL,
                max_tokens=50,
                messages=[
                    {"role": "user", "content": "Reply with just 'OK' to confirm connection."}
                ]
            )

            if response.content:
                return True, "Connection successful!"
            else:
                return False, "Empty response from API"

        except anthropic.AuthenticationError:
            return False, "Invalid API key"
        except anthropic.RateLimitError:
            return False, "Rate limit exceeded - but key is valid"
        except anthropic.APIError as e:
            return False, f"API error: {str(e)}"
        except Exception as e:
            return False, f"Connection failed: {str(e)}"

    def _load_image_as_base64(self, image_path: str) -> Tuple[str, str]:
        """
        Load an image file and convert to base64.

        Args:
            image_path: Path to the image file

        Returns:
            Tuple of (base64_data, media_type)
        """
        path = Path(image_path)

        # Determine media type from extension
        ext = path.suffix.lower()
        media_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp',
        }

        media_type = media_types.get(ext, 'image/jpeg')

        # Read and encode
        with open(path, 'rb') as f:
            image_data = base64.standard_b64encode(f.read()).decode('utf-8')

        return image_data, media_type

    def describe_image(
        self,
        image_path: str,
        prompt: str,
        max_tokens: int = 500
    ) -> Tuple[bool, str]:
        """
        Generate a description for an image using Claude's vision.

        Args:
            image_path: Path to the image file
            prompt: Instruction prompt for description style
            max_tokens: Maximum response length

        Returns:
            Tuple of (success: bool, description_or_error: str)
        """
        if not self.client:
            return False, "API client not initialized"

        try:
            # Load image
            image_data, media_type = self._load_image_as_base64(image_path)

            # Build message with image
            response = self.client.messages.create(
                model=self.DEFAULT_MODEL,
                max_tokens=max_tokens,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": image_data,
                                },
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ],
                    }
                ],
            )

            if response.content:
                description = response.content[0].text.strip()
                return True, description
            else:
                return False, "Empty response from API"

        except FileNotFoundError:
            return False, f"Image not found: {image_path}"
        except anthropic.APIError as e:
            return False, f"API error: {str(e)}"
        except Exception as e:
            return False, f"Error: {str(e)}"

    def analyze_training(
        self,
        metrics_summary: Dict[str, Any],
        graph_image_bytes: Optional[bytes],
        high_loss_images: List[Dict[str, Any]],
        current_status: Dict[str, Any],
    ) -> Tuple[bool, str]:
        """
        Analyze a training run and provide suggestions.

        Args:
            metrics_summary: Summary statistics from metrics.jsonl
            graph_image_bytes: PNG bytes of the loss graph (optional)
            high_loss_images: List of high-loss image info dicts
            current_status: Current training status dict

        Returns:
            Tuple of (success: bool, analysis_or_error: str)
        """
        if not self.client:
            return False, "API client not initialized"

        try:
            # Build the analysis prompt
            prompt_parts = [
                "You are an expert at analyzing LoRA training runs for Flux image generation models.",
                "",
                "## Current Training Status:",
                f"- Current Step: {current_status.get('current_step', 'N/A')} / {current_status.get('total_steps', 'N/A')}",
                f"- Progress: {current_status.get('progress_percent', 0):.1f}%",
                f"- Current Loss: {current_status.get('loss', 'N/A')}",
                f"- Trend: {current_status.get('trend_status', 'unknown')}",
                "",
            ]

            # Add metrics summary
            if metrics_summary:
                prompt_parts.extend([
                    "## Training Metrics Summary:",
                    f"- Total steps recorded: {metrics_summary.get('total_steps', 'N/A')}",
                    f"- Average loss: {metrics_summary.get('avg_loss', 'N/A'):.4f}" if metrics_summary.get('avg_loss') else "- Average loss: N/A",
                    f"- Min loss: {metrics_summary.get('min_loss', 'N/A'):.4f}" if metrics_summary.get('min_loss') else "- Min loss: N/A",
                    f"- Max loss: {metrics_summary.get('max_loss', 'N/A'):.4f}" if metrics_summary.get('max_loss') else "- Max loss: N/A",
                    f"- EMA-10: {metrics_summary.get('ema_10', 'N/A')}",
                    f"- EMA-50: {metrics_summary.get('ema_50', 'N/A')}",
                    f"- EMA-100: {metrics_summary.get('ema_100', 'N/A')}",
                    f"- Trend slope: {metrics_summary.get('trend_slope', 'N/A')}",
                    "",
                ])

            # Add high-loss images info
            if high_loss_images:
                prompt_parts.extend([
                    "## High-Loss Images (potential problems):",
                ])
                for img in high_loss_images[:10]:  # Limit to top 10
                    z_score = img.get('z_score', 0)
                    outlier_marker = "[OUTLIER] " if z_score > 0 else ""
                    prompt_parts.append(
                        f"- {outlier_marker}{img.get('filename', 'Unknown')}: "
                        f"mean_loss={img.get('mean_loss', 0):.4f}"
                    )
                prompt_parts.append("")

            # Add analysis request
            prompt_parts.extend([
                "## Please analyze this training run and provide:",
                "1. **Overall Assessment**: Is training progressing well? Rate it (Good/Okay/Concerning).",
                "2. **Loss Curve Analysis**: What does the trend tell us?",
                "3. **High-Loss Images**: What might these indicate about the dataset?",
                "4. **Specific Recommendations**: Actionable steps to improve training results.",
                "5. **When to Stop**: Based on the trend, when should training be stopped?",
                "",
                "Be concise and actionable. Focus on practical advice.",
            ])

            prompt_text = "\n".join(prompt_parts)

            # Build message content
            content = []

            # Add graph image if available
            if graph_image_bytes:
                graph_base64 = base64.standard_b64encode(graph_image_bytes).decode('utf-8')
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": graph_base64,
                    },
                })
                content.append({
                    "type": "text",
                    "text": "Above is the training loss graph.\n\n" + prompt_text
                })
            else:
                content.append({
                    "type": "text",
                    "text": prompt_text
                })

            # Make API call
            response = self.client.messages.create(
                model=self.DEFAULT_MODEL,
                max_tokens=1500,
                messages=[
                    {"role": "user", "content": content}
                ],
            )

            if response.content:
                analysis = response.content[0].text.strip()
                return True, analysis
            else:
                return False, "Empty response from API"

        except anthropic.APIError as e:
            return False, f"API error: {str(e)}"
        except Exception as e:
            return False, f"Analysis failed: {str(e)}"


def get_api_status() -> str:
    """Check if anthropic package is available."""
    if ANTHROPIC_AVAILABLE:
        return "anthropic package installed"
    else:
        return "anthropic package NOT installed - run: pip install anthropic"
