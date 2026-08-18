"""VectorForge — CPU ANN retrieval engine.

Phase 0/1 exposes exact FlatIndex only. HNSW is intentionally not imported yet.
"""

from vectorforge._vectorforge import FlatIndex, __version__

__all__ = ["FlatIndex", "__version__"]
