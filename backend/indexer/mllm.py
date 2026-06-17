"""
MLLM client for generating natural-language cluster descriptions.

Calls a locally-running vLLM server (OpenAI-compatible API) with
representative frame thumbnails and cluster statistics to produce
one-sentence cluster descriptions and a video-level meta-summary.

Usage:
    client = MLLMClient(config.mllm)
    if client.is_available():
        desc = client.describe_cluster(images_b64, stats)
        summary = client.generate_meta_summary(descriptions, n_clusters)
"""

from __future__ import annotations

from typing import Dict, List, Optional

import httpx
from loguru import logger
from openai import OpenAI

from indexer.config import MLLMConfig


class MLLMClient:
    def __init__(self, config: MLLMConfig):
        self.config = config
        self._client: Optional[OpenAI] = None

    def _get_client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(
                base_url=self.config.base_url,
                api_key="dummy",
                timeout=self.config.timeout_s,
            )
        return self._client

    def is_available(self) -> bool:
        """Quick ping to the vLLM /models endpoint."""
        if not self.config.enabled:
            return False
        try:
            resp = httpx.get(
                f"{self.config.base_url}/models",
                timeout=5.0,
            )
            return resp.status_code == 200
        except Exception:
            return False

    def describe_cluster(
        self,
        images_b64: List[str],
        stats: Dict,
    ) -> Optional[str]:
        """
        Generate a one-sentence description for a cluster.

        Args:
            images_b64: Base64 JPEG strings for up to 5 representative frames.
            stats: dict with keys member_count (int), class_dist (Dict[str, float]),
                   avg_speed (float).

        Returns:
            Description string, or None on failure.
        """
        if not self.config.enabled or not images_b64:
            return None
        try:
            content: list = []
            for b64 in images_b64[:5]:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                })

            class_dist_str = ", ".join(
                f"{cls} {pct:.0f}%"
                for cls, pct in sorted(
                    stats.get("class_dist", {}).items(),
                    key=lambda kv: -kv[1],
                )
            )
            content.append({
                "type": "text",
                "text": (
                    "These are representative frames from a surveillance video cluster.\n"
                    "Write ONE sentence (20 words or fewer) describing what activity "
                    "or object type this cluster represents. Focus on the object or activity attributes."
                ),
            })

            resp = self._get_client().chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": content}],
                max_tokens=self.config.max_tokens,
            )
            return resp.choices[0].message.content.strip()
        except Exception as exc:
            logger.warning(f"MLLM describe_cluster failed: {exc}")
            return None

    def generate_meta_summary(
        self,
        descriptions: List[str],
        n_clusters: int,
    ) -> Optional[str]:
        """
        Generate a 1-2 sentence meta-summary explaining what the clusters distinguish.

        Args:
            descriptions: One description string per cluster (e.g. "Cluster 0: ...").
            n_clusters: Total number of clusters.

        Returns:
            Meta-summary string, or None on failure.
        """
        if not self.config.enabled or len(descriptions) < 2:
            return None
        try:
            bullet_list = "\n".join(f"- {d}" for d in descriptions)
            resp = self._get_client().chat.completions.create(
                model=self.config.model,
                messages=[{
                    "role": "user",
                    "content": (
                        f"A surveillance video was clustered into {n_clusters} groups. "
                        f"Here are their descriptions:\n{bullet_list}\n\n"
                        "Write 1-2 sentences explaining what these clusters collectively "
                        "distinguish from each other (e.g. different object types, "
                        "speeds, or movement patterns)."
                    ),
                }],
                max_tokens=80,
            )
            return resp.choices[0].message.content.strip()
        except Exception as exc:
            logger.warning(f"MLLM generate_meta_summary failed: {exc}")
            return None
