"""
Feature cache placeholder to satisfy imports.
"""


class FeatureCache:
    """No-op cache placeholder."""

    def get(self, key):
        return None

    def set(self, key, value, ttl_seconds=None):
        return None


def get_feature_cache() -> FeatureCache:
    return FeatureCache()
