"""Provider registry. Selecting a backend happens here and nowhere else."""

import os

from backend.core.integrations.base import ServiceDeskProvider
from backend.core.integrations.rest import GenericRestProvider

_providers: dict[str, type[ServiceDeskProvider]] = {
    # The generic REST adapter covers Jira/ServiceNow/anything with a JSON API;
    # add a key here (e.g. "jira": JiraProvider) for a system needing a
    # purpose-built subclass.
    "rest": GenericRestProvider,
}


def get_provider() -> ServiceDeskProvider:
    name = os.getenv("SERVICEDESK_PROVIDER", "rest")
    if name not in _providers:
        raise ValueError(
            f"Unknown SERVICEDESK_PROVIDER {name!r}. Available: {sorted(_providers)}"
        )
    return _providers[name]()
