import os
import time
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types

# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="Financial Copilot AI",
    page_icon="📊",
    layout="wide"
)

# =====================================================
# 2. LOAD GEMINI API KEY
# =====================================================
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

try:
    if "GEMINI_API_KEY" in st.secrets:
        GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except Exception:
    pass

if not GEMINI_API_KEY:
    st.error(
        "🚨 Gemini API Key is missing! "
        "Set GEMINI_API_KEY in Streamlit Secrets or a .env file."
    )
    st.stop()

# =====================================================
# 3. INITIALIZE GEMINI CLIENT & RETRY HELPER
# =====================================================
client = genai.Client(api_key=GEMINI_API_KEY)


def ask_gemini_with_retry(model_name, prompt_text, sys_instruction, retries=3, delay=4):
    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt_text,
                config=types.GenerateContentConfig(
                    system_instruction=sys_instruction
                ),
            )
            return response.text
        except Exception as e:
            err_msg = str(e)
            if ("503" in err_msg or "UNAVAILABLE" in err_msg) and attempt < retries - 1:
                time.sleep(delay)
            else:
                raise e


# =====================================================
# 4. STREAMLIT APP UI
# =====================================================
st.title("📊 Financial Copilot AI – Scenario Planning & Strategic Insights")
st.write("Upload financial data and enter a scenario prompt to simulate different projections!")

# Model selector (Google Gemini)
selected_model = st.selectbox(
    "🤖 Select AI Model",
    ["gemini-3.8-flash", "gemini-3.8-pro"],
    index=0
)

# =====================================================
# 5. FILE UPLOADER
# =====================================================
uploaded_file = st.file_uploader("📂 Upload your dataset (Excel format)", type=["xlsx"])

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)

        # Check for required columns
        required_columns = ["Category", "Base Forecast"]
        if not all(col in df.columns for col in required_columns):
            st.error("⚠️ The uploaded file must contain 'Category' and 'Base Forecast' columns!")
            st.stop()

        df["Base Forecast"] = pd.to_numeric(df["Base Forecast"], errors="coerce")
        if df["Base Forecast"].isna().any():
            st.error("⚠️ 'Base Forecast' column contains invalid numeric values.")
            st.stop()

        # Scenario Input
        scenario_prompt = st.text_area(
            "📝 Enter a financial scenario (e.g., 'Revenue drops 10%', 'Costs increase by 5%'):"
        )

        if st.button("🚀 Generate Scenarios"):
            with st.spinner("Simulating scenarios and generating Gemini insights..."):
                df_calc = df.copy()
                df_calc["Optimistic"] = df_calc["Base Forecast"] * np.random.uniform(1.1, 1.3, len(df_calc))
                df_calc["Pessimistic"] = df_calc["Base Forecast"] * np.random.uniform(0.7, 0.9, len(df_calc))
                df_calc["Worst Case"] = df_calc["Base Forecast"] * np.random.uniform(0.5, 0.7, len(df_calc))

                st.session_state.scenario_df = df_calc
                st.session_state.active_scenario = scenario_prompt
                st.session_state.chat_history = []

                # Limit preview to 20 rows to avoid token overload
                df_preview = df_calc.head(20).to_string(index=False)

                sys_prompt = (
                    "You are an AI Financial Copilot. "
                    "You help analyze scenario planning, variance analysis, and strategy. "
                    "You can discuss EBIT, revenue, margin, COGS, OPEX, CAPEX, cash flow, "
                    "liquidity, financial distress, firm value, ESG, risk, and growth strategy. "
                    "Always provide structured insights with both short-term and long-term recommendations."
                )
                user_msg = (
                    f"Here are the scenario projections:\n{df_preview}\n"
                    f"Scenario: {scenario_prompt}\n"
                    "Please summarize the key insights and recommendations."
                )

                try:
                    analysis_text = ask_gemini_with_retry(selected_model, user_msg, sys_prompt)
                    st.session_state.initial_analysis = analysis_text
                except Exception as e:
                    st.session_state.initial_analysis = f"⚠️ AI request failed: {e}"

        # =================================================
        # 6. TWO-COLUMN LAYOUT (DATA/CHART & AI COPILOT)
        # =================================================
        if "scenario_df" in st.session_state:
            df_result = st.session_state.scenario_df
            saved_scenario = st.session_state.get("active_scenario", "")

            col1, col2 = st.columns([2, 1])

            with col1:
                st.subheader("📊 Scenario-Based Projections")
                st.dataframe(df_result, use_container_width=True)

                fig_scenarios = px.bar(
                    df_result,
                    x="Category",
                    y=["Base Forecast", "Optimistic", "Pessimistic", "Worst Case"],
                    title="📉 Scenario Planning: Financial Projections",
                    barmode="group",
                    text_auto=".2s",
                )
                st.plotly_chart(fig_scenarios, use_container_width=True)

            with col2:
                st.subheader("🤖 AI Financial Copilot Insights")
                st.markdown("**AI Initial Analysis:**")
                st.write(st.session_state.get("initial_analysis", ""))

                if "chat_history" not in st.session_state:
                    st.session_state.chat_history = []

                user_query = st.text_input(
                    "💬 Ask your Financial Copilot anything (EBIT, Cash Flow, CAPEX, ESG, etc.)"
                )
                col_chat1, col_chat2 = st.columns([4, 1])
                with col_chat1:
                    send_btn = st.button("Send")
                with col_chat2:
                    reset_btn = st.button("🔄 Reset Chat")

                if reset_btn:
                    st.session_state.chat_history = []
                    st.success("Chat history cleared!")

                if send_btn and user_query:
                    with st.spinner("Gemini is thinking..."):
                        df_preview = df_result.head(20).to_string(index=False)
                        history_text = "\n".join(
                            [f"{m['role'].upper()}: {m['content']}" for m in st.session_state.chat_history]
                        )
                        full_prompt = (
                            f"Dataset preview:\n{df_preview}\n"
                            f"Scenario: {saved_scenario}\n\n"
                            f"Previous Conversation:\n{history_text}\n\n"
                            f"Current Question: {user_query}"
                        )
                        chat_sys_prompt = (
                            "You are an AI Financial Copilot. "
                            "You can answer questions about EBIT, revenue, margin, OPEX, CAPEX, cash flow, "
                            "firm value, ESG, risks, financial distress, and strategy. "
                            "Always explain clearly and give actionable recommendations."
                        )
                        try:
                            ai_answer = ask_gemini_with_retry(selected_model, full_prompt, chat_sys_prompt)
                            st.session_state.chat_history.append({"role": "user", "content": user_query})
                            st.session_state.chat_history.append({"role": "assistant", "content": ai_answer})
                        except Exception as e:
                            st.error(f"⚠️ AI chat request failed: {e}")

                if st.session_state.chat_history:
                    for msg in st.session_state.chat_history:
                        if msg["role"] == "user":
                            st.markdown(f"**👤 You:** {msg['content']}")
                        else:
                            st.markdown(f"**🤖 Copilot:** {msg['content']}")

    except Exception as e:
        st.error(f"Error processing uploaded file: {str(e)}")
