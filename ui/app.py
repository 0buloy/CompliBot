"""CompliBot demo UI.

    streamlit run ui/app.py                                    # in-process (loads the local index)
    COMPLIBOT_API_URL=http://localhost:8000 streamlit run ui/app.py   # talk to a running API

In API mode the role comes from the API key you enter, exactly like any other client.
In-process mode is for local demos and offers a role toggle instead.
"""

from __future__ import annotations

import json
import os

import streamlit as st

API_URL = os.environ.get("COMPLIBOT_API_URL", "").rstrip("/")

STATUS_BADGES = {
    "answered": ("Answered with verified citations", "green"),
    "refused": ("Not covered by the documents", "orange"),
    "unsupported": ("Answer NOT fully supported by retrieved clauses", "red"),
    "blocked": ("Blocked by input guardrail", "red"),
    "error": ("Error / degraded mode", "red"),
}

EXAMPLES = [
    "Can I paste a customer's NRIC number into an AI tool?",
    "Make a checklist for reporting AI incidents and handling customer data",
    "Who has to approve AI-generated messages before they go to customers?",
    "What is the maximum fine for breaching the AI usage policy?",
]


@st.cache_resource(show_spinner="Loading index and guardrails...")
def local_bot():
    from complibot.answer import CompliBot
    from complibot.config import Settings

    return CompliBot.from_settings(Settings.from_env())


def call(mode: str, question: str, role: str, api_key: str) -> dict:
    if API_URL:
        import httpx

        headers = {"X-API-Key": api_key} if api_key else {}
        resp = httpx.post(f"{API_URL}/{mode}", json={"question": question}, headers=headers, timeout=120)
        resp.raise_for_status()
        return resp.json()
    bot = local_bot()
    if mode == "agent":
        from complibot.agent import Agent

        return Agent.from_bot(bot).run(question, role=role).model_dump()
    return bot.ask(question, role=role).model_dump()


def render(result: dict) -> None:
    label, color = STATUS_BADGES.get(result["status"], (result["status"], "gray"))
    st.markdown(f":{color}-badge[{label}]")
    if result.get("pii_redacted"):
        found = ", ".join(f"{n} {k}" for k, n in result["pii_redacted"].items())
        st.info(f"PII redacted before reaching the LLM: {found}. The model saw: _{result['question']}_")
    st.markdown(result["answer"])

    if result.get("checklist"):
        st.subheader("Checklist")
        rows = [{"#": i["id"], "Requirement": i["requirement"], "Priority": i["priority"],
                 "Source": f"{i['source']['doc_title']} s.{i['source']['section']} (p.{i['source']['page']})"}
                for i in result["checklist"]["items"]]
        st.dataframe(rows, hide_index=True, use_container_width=True)
        st.download_button("Download checklist JSON", json.dumps(result["checklist"], indent=2),
                           file_name="checklist.json", mime="application/json")

    if result.get("citations"):
        st.subheader("Sources")
        for c in result["citations"]:
            with st.expander(f"{c['doc_title']} - s.{c['section_id']} {c['section_title']} (p.{c['page_start']})"):
                st.write(c["snippet"])
    if result.get("related_clauses"):
        st.subheader("Most relevant clauses (LLM unavailable)")
        for c in result["related_clauses"]:
            st.write(f"- {c['doc_title']} s.{c['section_id']}: {c['snippet']}")
    if result.get("invalid_citations"):
        st.warning(f"The model cited sources that were not retrieved: {result['invalid_citations']}")

    if result.get("steps"):
        with st.expander(f"Agent trace ({len(result['steps'])} tool calls)"):
            for i, s in enumerate(result["steps"], start=1):
                st.code(f"{i}. {s['tool']}({json.dumps(s['args'])})\n   -> {'ok' if s['ok'] else 'ERROR'}: "
                        f"{s['summary']}  [{s['latency_ms']:.0f} ms]", language="text")
    u = result.get("usage", {})
    st.caption(f"role={result['role']} · latency {result['latency_ms']:.0f} ms · tokens "
               f"{u.get('input_tokens', 0)}/{u.get('output_tokens', 0)} · ${u.get('cost_usd', 0):.4f} · "
               f"model {u.get('model') or '-'}")


def main() -> None:
    st.set_page_config(page_title="CompliBot", layout="centered")
    st.title("CompliBot")
    st.caption("Cited answers over Singapore regulatory documents (MAS · PDPC · IMDA). Not legal advice.")

    with st.sidebar:
        mode = st.radio("Mode", ["ask", "agent"], format_func=lambda m: {"ask": "Ask (RAG)",
                                                                          "agent": "Agent (tools)"}[m])
        if API_URL:
            st.write(f"API: `{API_URL}`")
            api_key = st.text_input("API key (internal access)", type="password")
            role = "public"
        else:
            api_key = ""
            role = st.radio("Role (local demo)", ["public", "internal"])
        st.divider()
        st.write("Try:")
        for ex in EXAMPLES:
            if st.button(ex, use_container_width=True):
                st.session_state["question"] = ex

    question = st.text_area("Your question", key="question", height=90,
                            placeholder="e.g. How quickly must AI incidents be reported?")
    if st.button("Submit", type="primary", disabled=not question.strip()):
        with st.spinner("Searching clauses..."):
            try:
                result = call(mode, question, role, api_key)
            except Exception as e:  # show a friendly error, never a stack trace
                st.error(f"Request failed: {e}")
                return
        render(result)


main()
