import os
from datetime import datetime
import pandas as pd
import streamlit as st
from groq import Groq
from openai import OpenAI

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Semicon AI Career Hub | Upside-Down Search",
    page_icon="🔬",
    layout="wide",
)

# --- MODEL REGISTRY (GROQ & AI SINGAPORE SEA-LION) ---
MODEL_REGISTRY = {
    "Sea-Lion v4 27B": {
        "provider": "ai_singapore",
        "model_id": "aisingapore/Gemma-SEA-LION-v4-27B-IT",
        "description": "Gemma 3 27B fine-tuned for SEA context by AI Singapore.",
    },
    "Sea-Lion v4 32B (Qwen)": {
        "provider": "ai_singapore",
        "model_id": "aisingapore/Qwen-SEA-LION-v4-32B-IT",
        "description": "Qwen3-32B fine-tuned for SEA tasks.",
    },
    "Groq: GPT-OSS 120B": {
        "provider": "groq",
        "model_id": "openai/gpt-oss-120b",
        "description": "OpenAI's open-weight MoE, 120B params, reasoning-capable.",
    },
    "Groq: GPT-OSS 20B": {
        "provider": "groq",
        "model_id": "openai/gpt-oss-20b",
        "description": "Smaller, faster OpenAI open-weight model.",
    },
    "Groq: Llama 3.3 70B": {
        "provider": "groq",
        "model_id": "llama-3.3-70b-versatile",
        "description": "Meta's Llama 3.3 70B hosted on Groq.",
    },
}

DEFAULT_MODEL_KEY = "Sea-Lion v4 27B"
AI_SINGAPORE_BASE_URL = "https://api.sea-lion.ai/v1"

# --- LOAD API KEYS ---
groq_api_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))
sealion_api_key = st.secrets.get("SEALION_API_KEY", os.environ.get("SEALION_API_KEY", ""))

# --- DATA PERSISTENCE (CSV BACKED FOR SIMPLE GITHUB DEPLOY) ---
CRM_FILE = "semicon_network_crm.csv"

DEFAULT_DATA = [
    {
        "Name": "Example: Dr. Tan Wei Ming",
        "Role": "Smart Manufacturing Director",
        "Company": "Wafer Fab / OSAT",
        "Pyramid_Layer": "Weak Tie (Former Colleague)",
        "Status": "To Contact",
        "Last_Touch": datetime.today().strftime("%Y-%m-%d"),
        "Notes": "Ask about PdM & FDC model adoption bottlenecks on the fab floor.",
    },
    {
        "Name": "Example: Sarah Lim",
        "Role": "Committee Lead (Smart Mfg)",
        "Company": "SSIA / SEMI SEA",
        "Pyramid_Layer": "Professional Body / Association",
        "Status": "Researching",
        "Last_Touch": datetime.today().strftime("%Y-%m-%d"),
        "Notes": "Speaking at upcoming Industry 4.0 roundtable; volunteer for working group.",
    },
    {
        "Name": "Example: Marcus Koh",
        "Role": "Delivery Director (OT/IT)",
        "Company": "Semicon System Integrator",
        "Pyramid_Layer": "Side Door (Contract / Fractional)",
        "Status": "To Contact",
        "Last_Touch": datetime.today().strftime("%Y-%m-%d"),
        "Notes": "Pitch 3-month AI BA/PM pilot governance support for fab MES/AOI rollout.",
    },
]


def load_crm() -> pd.DataFrame:
    if os.path.exists(CRM_FILE):
        return pd.read_csv(CRM_FILE)
    df = pd.DataFrame(DEFAULT_DATA)
    df.to_csv(CRM_FILE, index=False)
    return df


def save_crm(df: pd.DataFrame):
    df.to_csv(CRM_FILE, index=False)


# --- SYSTEM PROMPT FOR AI COPILOT ---
SYSTEM_INSTRUCTION = """
You are an executive career strategist specializing in Semiconductor Digital Transformation 
and AI Process Transformation (Yield Optimization, Predictive Maintenance, AOI, FDC, SPC, MES/ERP integration) 
in Singapore and Southeast Asia.

You strictly follow the 'Upside-Down Hiring Pyramid' philosophy:
1. Never write generic job-application cover letters.
2. Focus on bottom-up trust-building: Weak Ties (former colleagues/acquaintances), Professional Bodies (SSIA, SEMI SEA, SCS), 
   and the Side Door (contract, fractional, or pilot-project roles via system integrators or fab CapEx teams).
3. Keep outreach messages concise (under 90 words), warm, low-pressure, and anchored on a specific semiconductor domain question—never asking directly for a job.
4. Position the candidate as a 'Bridge-Builder' who translates process engineering constraints into reliable, IP-safe AI deployments.
"""


# --- UNIFIED LLM CALLER (GROQ + AI SINGAPORE) ---
def call_llm(prompt: str, model_key: str, groq_key: str, sealion_key: str) -> str:
    entry = MODEL_REGISTRY.get(model_key, MODEL_REGISTRY[DEFAULT_MODEL_KEY])
    provider = entry["provider"]
    model_id = entry["model_id"]

    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": prompt},
    ]

    if provider == "groq":
        if not groq_key:
            return "⚠️ Please enter your `GROQ_API_KEY` in the sidebar (or `.streamlit/secrets.toml`)."
        try:
            client = Groq(api_key=groq_key)
            resp = client.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=0.7,
            )
            return resp.choices[0].message.content
        except Exception as e:
            return f"❌ Groq API Error ({model_id}): {str(e)}"

    elif provider == "ai_singapore":
        if not sealion_key:
            return "⚠️ Please enter your `SEALION_API_KEY` in the sidebar (or `.streamlit/secrets.toml`)."
        try:
            client = OpenAI(api_key=sealion_key, base_url=AI_SINGAPORE_BASE_URL)
            resp = client.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=0.7,
            )
            return resp.choices[0].message.content
        except Exception as e:
            return f"❌ AI Singapore API Error ({model_id}): {str(e)}"

    return "❌ Unknown provider selected."


# --- SIDEBAR ---
with st.sidebar:
    st.title("🔬 Semicon AI Hub")
    st.caption("Bottom-Up Career Engine for AI BA / PM")

    st.subheader("🧠 AI Model Selection")
    selected_model_key = st.selectbox(
        "Choose Model",
        options=list(MODEL_REGISTRY.keys()),
        index=list(MODEL_REGISTRY.keys()).index(DEFAULT_MODEL_KEY),
    )
    active_entry = MODEL_REGISTRY[selected_model_key]
    st.caption(f"Provider: **{active_entry['provider']}** · `{active_entry['model_id']}`")
    st.caption(active_entry["description"])

    # API Key inputs if not already set in secrets/env
    if active_entry["provider"] == "ai_singapore":
        if not sealion_api_key:
            sealion_api_key = st.text_input("SEA-LION API Key", type="password")
        else:
            st.success("SEA-LION API Key loaded.")
    elif active_entry["provider"] == "groq":
        if not groq_api_key:
            groq_api_key = st.text_input("Groq API Key", type="password")
        else:
            st.success("Groq API Key loaded.")

    st.divider()
    st.subheader("🎯 Target Positioning")
    target_role = st.selectbox(
        "Primary Target Role",
        [
            "AI Business Analyst (Semicon / Smart Mfg)",
            "AI Project Manager (Fab Digital Transformation)",
            "Fractional / Contract AI Delivery Lead",
        ],
    )
    user_background = st.text_area(
        "Your Core Edge (1–2 lines)",
        value="Bridging process/manufacturing workflows and AI lifecycle delivery (requirements, data readiness, stakeholder adoption).",
        height=90,
    )

    st.divider()
    st.markdown(
        """
        **Upside-Down Priority:**
        1. 🚪 Side Door (Temps / Consultants)
        2. 🤝 Weak Ties & Staff Contacts
        3. 🏛️ Associations (SSIA / SEMI / SCS)
        4. 🚫 Avoid Public Job Boards
        """
    )

# --- MAIN HEADER & METRICS ---
st.title("Semiconductor AI Transformation — Career Command Center")
df_crm = load_crm()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Pipeline Contacts", len(df_crm))
col2.metric(
    "Weak Ties & Staff",
    len(df_crm[df_crm["Pyramid_Layer"].str.contains("Weak Tie", na=False)]),
)
col3.metric(
    "Associations (SSIA/SEMI/SCS)",
    len(df_crm[df_crm["Pyramid_Layer"].str.contains("Association", na=False)]),
)
col4.metric(
    "Side-Door Targets",
    len(df_crm[df_crm["Pyramid_Layer"].str.contains("Side Door", na=False)]),
)

# --- TABS ---
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📊 1. Bottom-Up CRM Tracker",
        "✍️ 2. Weak-Tie & Association Outreach",
        "🚪 3. Side-Door Pilot Pitch & Intel",
        "🗓️ 4. 14-Day Sprint Checklist",
    ]
)

# === TAB 1: CRM TRACKER ===
with tab1:
    st.subheader("Bottom-Up Relationship Pipeline")
    st.caption("Edit cells directly below or add new rows. Click 'Save Changes' to persist.")

    edited_df = st.data_editor(
        df_crm,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "Pyramid_Layer": st.column_config.SelectboxColumn(
                "Pyramid Layer",
                options=[
                    "Side Door (Contract / Fractional)",
                    "Weak Tie (Former Colleague)",
                    "Professional Body / Association",
                    "Fab / Equipment Vendor Lead",
                ],
                required=True,
            ),
            "Status": st.column_config.SelectboxColumn(
                "Status",
                options=[
                    "Researching",
                    "To Contact",
                    "Message Sent",
                    "Chat Booked",
                    "Referral / Pilot Active",
                ],
                required=True,
            ),
        },
    )

    if st.button("💾 Save CRM Changes", type="primary"):
        save_crm(edited_df)
        st.success("Pipeline saved!")

# === TAB 2: OUTREACH GENERATOR ===
with tab2:
    st.subheader("Zero-Pressure Outreach Generator")
    c1, c2 = st.columns(2)

    with c1:
        outreach_type = st.radio(
            "Outreach Channel",
            [
                "Weak Tie Reconnection (Former Colleague / Alumni)",
                "Association Speaker / Committee Lead (SSIA, SEMI SEA, SCS)",
            ],
        )
        contact_name = st.text_input("Contact Name", placeholder="e.g., David Lim")
        contact_company = st.text_input("Company / Association", placeholder="e.g., Micron / SSIA")
        contact_context = st.text_area(
            "Context / LinkedIn Bio / Panel Topic",
            placeholder="Paste their LinkedIn summary, past shared project, or the panel topic they spoke on...",
            height=130,
        )
        generate_outreach_btn = st.button("✨ Generate Outreach Message", type="primary")

    with c2:
        st.markdown(f"**Generated Message via {selected_model_key}**")
        if generate_outreach_btn:
            prompt = f"""
            Create a LinkedIn/email outreach message for:
            - Channel: {outreach_type}
            - Contact: {contact_name} at {contact_company}
            - Context: {contact_context}
            - My Target Role: {target_role}
            - My Edge: {user_background}

            Provide:
            1. A concise message (< 90 words) asking ONE smart domain question (no asking for a job).
            2. A 1-line follow-up question to use during the 10-minute chat to uncover AI pilot bottlenecks.
            """
            with st.spinner(f"Drafting high-trust outreach with {selected_model_key}..."):
                result = call_llm(prompt, selected_model_key, groq_api_key, sealion_api_key)
                st.markdown(result)

# === TAB 3: SIDE-DOOR PITCH & COMPANY INTEL ===
with tab3:
    st.subheader("Side-Door Pilot Pitch & Fab Pain-Point Analyzer")
    st.caption(
        "Target boutique semicon system integrators, equipment vendors, or fab smart manufacturing leads with a concrete pilot proposal."
    )

    company_target = st.text_input(
        "Target Fab, Equipment Maker, or System Integrator",
        placeholder="e.g., GlobalFoundries Singapore, Applied Materials, or a local OT/IT integrator",
    )
    company_notes = st.text_area(
        "Paste Recent News, Annual Report Excerpt, or Service Page",
        placeholder="Paste text about their smart manufacturing, yield, packaging, or automation push...",
        height=130,
    )

    col_a, col_b = st.columns(2)
    with col_a:
        intel_btn = st.button("🔍 Analyze Fab AI Pain Points")
    with col_b:
        pitch_btn = st.button("🚪 Generate 3-Month Side-Door Pilot Pitch", type="primary")

    if intel_btn and company_target:
        intel_prompt = f"""
        Analyze {company_target} in the context of semiconductor digital transformation.
        Notes provided: {company_notes}
        Identify:
        1. Top 3 likely AI/data bottlenecks (e.g., sensor telemetry hygiene, process engineer trust, MES integration, HBM/advanced packaging inspection).
        2. 3 sharp questions an {target_role} should ask their engineering/delivery leads.
        """
        with st.spinner(f"Analyzing semiconductor pain points via {selected_model_key}..."):
            st.markdown(call_llm(intel_prompt, selected_model_key, groq_api_key, sealion_api_key))

    if pitch_btn and company_target:
        pitch_prompt = f"""
        Draft a 150-word 'Side-Door' cold email to a Delivery Director or Smart Manufacturing Lead at {company_target}.
        Notes: {company_notes}
        My Role: {target_role}
        My Edge: {user_background}
        Propose a low-risk, 3-month contract/fractional engagement to unblock stalled AI pilots, gather process requirements, and bridge data science with fab floor engineers.
        """
        with st.spinner(f"Drafting side-door pilot pitch via {selected_model_key}..."):
            st.markdown(call_llm(pitch_prompt, selected_model_key, groq_api_key, sealion_api_key))

# === TAB 4: 14-DAY EXECUTION TRACKER ===
with tab4:
    st.subheader("14-Day Upside-Down Execution Sprint")
    tasks = [
        "Day 1–2: Rewrite LinkedIn headline & CV around Semicon-to-AI translation (Yield, PdM, FDC, AOI).",
        "Day 3: Add 15–20 weak ties (past colleagues, vendor contacts, alumni) into Tab 1 CRM.",
        "Day 4–5: Send 20 zero-pressure curiosity messages using Tab 2.",
        "Day 6–7: Draft a 1-page 'AI Pilot Governance Blueprint' for manufacturing environments.",
        "Day 8–10: Conduct 3–5 informational chats; ask about fab floor AI adoption bottlenecks.",
        "Day 11–12: Pitch 5 semicon system integrators / consultancies for contract BA/PM roles (Tab 3).",
        "Day 13: Register for an SSIA, SEMI SEA, or SCS Smart Manufacturing event & message 2 speakers.",
        "Day 14: Review conversion rates in Tab 1 and double down on highest-responding channel.",
    ]
    for task in tasks:
        st.checkbox(task)
