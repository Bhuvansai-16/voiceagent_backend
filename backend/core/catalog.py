"""Measured model facts. Data, not a request handler.

Lived inside the debug router until the split. MODEL_PROVIDER says which
registered provider runs a model; MODEL_NOTES carries measurements no API
can report. Kept separate from providers.py on purpose: the registry says
what is buildable, this says what has been measured.
"""

# Measured by hand from a bench, not fetched from any provider. An API listing
# cannot tell you that gemini-2.5-flash-lite writes without asking 3 times out
# of 3, and picking a model by name alone is how that got chosen once. The
# registry says what is buildable; this says what has been measured. Keep them
# separate: a hand-entered figure and a measured distribution are different
# kinds of claim, and /_debug/analytics reports the third kind.
MODEL_NOTES: dict[str, dict] = {
    "gemini-3.5-flash-lite": {"ms": 695, "safe": True,
        "note": "695ms first token, asked before writing 3/3. Default."},
    "gemini-3.1-flash-lite": {"ms": 692, "safe": None, "note": "692ms, only probed once."},
    "gemini-2.5-flash": {"ms": 1225, "safe": True,
        "note": "1121-1328ms, asked before writing 3/3."},
    "gemini-3.6-flash": {"ms": 1894, "safe": None, "note": "1894ms, only probed once."},
    "gemini-3.5-flash": {"ms": 6797, "safe": True,
        "note": "6797ms. Safe, and too slow for speech."},
    "gemini-2.5-flash-lite": {"ms": None, "safe": False,
        "note": "Wrote without asking 3/3. Do not use for writes."},
    "nvidia/nvidia-nemotron-nano-9b-v2": {"ms": 739, "safe": None, "supports_thinking": True,
        "note": "739ms plain, 894ms with a tool call, tool called correctly 1/1 on a "
                "direct probe. Reasoning off via /no_think. Confirm-before-writing "
                "not yet seen on a real call."},
    "nvidia/nemotron-3.5-lightning-30b-a3b": {"ms": None, "safe": None,
        "supports_thinking": True, "note": "NVIDIA Nemotron 3.5 Lightning 30B A3B."},
    "nvidia/nvidia-nemotron-3.5-lightning-30b-a3b": {"ms": None, "safe": None,
        "supports_thinking": True,
        "note": "NVIDIA Nemotron 3.5 Lightning 30B A3B (build.nvidia.com alias)."},
    "deepseek-ai/deepseek-v4-pro-0813": {"ms": None, "safe": None, "supports_thinking": True,
        "note": "DeepSeek V4 Pro hosted by NVIDIA NIM."},
    "nvidia/Nemotron-3_5-Lightning": {"ms": None, "safe": None, "supports_thinking": True,
        "note": "NVIDIA Nemotron 3.5 Lightning 30B A3B on Nebius Token Factory."},
}

# Which registered provider each measured model runs on. Every value must exist
# in the registry; test_every_offered_model_names_a_registered_provider enforces
# it, so the picker cannot offer a model the worker would fail to build.
MODEL_PROVIDER: dict[str, str] = {
    "gemini-3.5-flash-lite": "google", "gemini-3.1-flash-lite": "google",
    "gemini-2.5-flash": "google", "gemini-3.6-flash": "google",
    "gemini-3.5-flash": "google", "gemini-2.5-flash-lite": "google",
    "nvidia/nvidia-nemotron-nano-9b-v2": "nvidia",
    "nvidia/nemotron-3.5-lightning-30b-a3b": "nvidia",
    "nvidia/nvidia-nemotron-3.5-lightning-30b-a3b": "nvidia",
    "deepseek-ai/deepseek-v4-pro-0813": "nvidia",
    "nvidia/Nemotron-3_5-Lightning": "nebius",
}
