"""Compiling a visual node graph into an agent configuration entry.

Pure: takes a flow dict, returns an agent entry dict. No I/O, no persona
reload, no HTTP. It lived inside the deploy route handler, where it could only
be exercised through a POST.

The one rule worth knowing: a flow that sets no speech nodes omits those keys
entirely rather than writing nulls, so deploying a graph with no voice nodes
cannot wipe an existing agent's STT/TTS configuration.
"""


def compile_flow(flow: dict, flow_id: str) -> dict:
    """Walk a flow's nodes and produce the agent entry they describe."""
    nodes = flow.get("nodes", [])
    flow_name = flow.get("name") or "Visual Flow Agent"
    flow_desc = flow.get("description") or ""

    agent_id = (
        (flow.get("deployed_agent_id") or flow_id.replace("flow-", ""))
        .strip().lower().replace(" ", "-")
    )
    agent_label = flow_name
    agent_blurb = flow_desc or f"Visual flow agent: {flow_name}"
    agent_purpose = "You are an AI voice assistant configured via Flow Studio."
    model_name = "gemini-3.5-flash-lite"
    provider_name = "google"
    voice_name = "aura-2-thalia-en"
    stt_provider_name = None
    stt_model_name = None
    tts_provider_name = None
    enable_thinking = False
    skills = []
    tools_set = set(["end_call"])
    toolkits_set = set()

    for node in nodes:
        ntype = node.get("type", "")
        ndata = node.get("data", {})
        nconfig = ndata.get("config", {})

        if ntype in ("agent_brain", "agent"):
            if nconfig.get("agent_label"):
                agent_label = nconfig["agent_label"]
            elif ndata.get("label"):
                agent_label = ndata["label"]
            if nconfig.get("blurb"):
                agent_blurb = nconfig["blurb"]
            if nconfig.get("purpose"):
                agent_purpose = nconfig["purpose"]
            if nconfig.get("skills"):
                skills = nconfig["skills"]

        elif ntype == "llm_engine":
            if nconfig.get("model"):
                model_name = nconfig["model"]
            if nconfig.get("provider"):
                provider_name = nconfig["provider"]
            if "enable_thinking" in nconfig:
                enable_thinking = bool(nconfig["enable_thinking"])

        elif ntype in ("voice_output", "tts_engine"):
            if nconfig.get("voice"):
                voice_name = nconfig["voice"]
            if nconfig.get("tts_provider"):
                tts_provider_name = nconfig["tts_provider"]

        elif ntype in ("voice_input", "trigger_voice", "stt_deepgram"):
            # The inspector writes the recogniser under `model`, matching the
            # shape the llm_engine node already uses.
            if nconfig.get("stt_provider"):
                stt_provider_name = nconfig["stt_provider"]
            if nconfig.get("model"):
                stt_model_name = nconfig["model"]
            # greeting is handled in the session prompt, not here.

        elif ntype in ("tool_ticket", "tool_password", "tool_websearch", "tool_sandbox",
                       "knowledge_rag", "memory_recall", "native_tool"):
            for t in nconfig.get("tools", []):
                tools_set.add(t)
            if ntype == "knowledge_rag":
                tools_set.add("search_documents")
            elif ntype == "memory_recall":
                tools_set.add("remember_about_caller")
            elif ntype == "tool_websearch":
                tools_set.add("search_web")
            elif ntype == "tool_sandbox":
                tools_set.add("run_command")

        elif ntype in ("mcp_composio", "connector_integration"):
            slug = (nconfig.get("slug") or nconfig.get("toolkit") or "").strip()
            if slug:
                toolkits_set.add(slug)

    agent_entry = {
        "id": agent_id,
        "label": agent_label,
        "blurb": agent_blurb,
        "purpose": agent_purpose,
        "model": model_name,
        "provider": provider_name,
        "voice": voice_name,
        "enable_thinking": enable_thinking,
        "tools": sorted(list(tools_set)),
        "toolkits": sorted(list(toolkits_set)),
        "skills": skills,
    }

    # Only what the flow actually set. A flow with no voice nodes must not
    # overwrite an existing agent's speech configuration with nulls.
    for key, value in (
        ("stt_provider", stt_provider_name),
        ("stt_model", stt_model_name),
        ("tts_provider", tts_provider_name),
    ):
        if value:
            agent_entry[key] = value

    return agent_entry
