import streamlit as st
from dotenv import load_dotenv
from supervisor_agent import fmcg_supervisor_agent
from langchain_core.globals import set_debug
from langchain_core.callbacks.stdout import StdOutCallbackHandler
from pdf_utils import build_pdf_bytes
from datetime import datetime

debug_handler = StdOutCallbackHandler()
set_debug(False)
load_dotenv()

st.set_page_config(
    page_title="FMCG Allocation Decision Agent",
    page_icon="📊",
)

st.title("📊 FMCG Allocation Decision Agent")

# Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_answer" not in st.session_state:
    st.session_state.last_answer = None

st.markdown(
    """
Aplikasi ini menggunakan **FMCG supervisor agent** untuk:
- Mengambil data makro per region (GDP, inflasi, fuel, sentiment, populasi, urbanisasi)
- Mengubahnya menjadi **Market Allocation Score (MAS)** per region
- Mengonversi MAS menjadi **proporsi alokasi**.

### Market Allocation Score (MAS)

MAS = (D * w_d) + (E * w_e) + (C * w_c)

- **D**: Demand Factor (0-1)  
- **E**: Economic Factor (0-1)  
- **C**: Cost Factor (0-1)  

Bobot **w_d, w_e, w_c** bisa kamu atur di panel.
"""
)

st.sidebar.header("Model Settings")

mode_label = st.sidebar.radio(
    "Model mode:",
    ["Thinking", "Standard"],
    index=1,  # default: non-thinking
)

if "model_mode" not in st.session_state:
    st.session_state["model_mode"] = "standard"

if "Thinking" in mode_label:
    st.session_state["model_mode"] = "thinking"
else:
    st.session_state["model_mode"] = "standard"

st.sidebar.header("⚙️ MAS Weight Settings")

wd_raw = st.sidebar.slider("Demand weight (w_d)", 0.0, 1.0, 0.4, 0.05)
we_raw = st.sidebar.slider("Economic weight (w_e)", 0.0, 1.0, 0.3, 0.05)
wc_raw = st.sidebar.slider("Cost weight (w_c)", 0.0, 1.0, 0.3, 0.05)
total_w = wd_raw + we_raw + wc_raw
if total_w == 0:
    wd, we, wc = 0.4, 0.3, 0.3
else:
    wd = wd_raw / total_w
    we = we_raw / total_w
    wc = wc_raw / total_w

st.sidebar.markdown(
    f"""
**Normalized weights (sum = 1):**
- w_d (demand): **{wd:.2f}**
- w_e (economic): **{we:.2f}**
- w_c (cost): **{wc:.2f}**
"""
)

st.markdown(
    """
Contoh prompt:

> *"Bandingkan potensi alokasi FMCG antara Jakarta, Malaysia, Singapore untuk 2025."*
"""
)

# Chat history

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


def extract_text_from_last_message(result_dict) -> str:
    if "messages" not in result_dict or not result_dict["messages"]:
        return "Tidak ada respon dari agent."

    last_msg = result_dict["messages"][-1]
    content = getattr(last_msg, "content", "")

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        texts = []
        for block in content:
            if isinstance(block, dict) and "text" in block:
                texts.append(block["text"])
            else:
                texts.append(str(block))
        return "\n".join(texts)

    return str(content)


user_input = st.chat_input("Tanya tentang alokasi FMCG di suatu negara/region...")

if user_input:
    if user_input.strip().lower().startswith("/savepdf"):
        if st.session_state.last_answer is None:
            with st.chat_message("assistant"):
                st.warning("Belum ada jawaban yang bisa disimpan ke PDF.")
        else:
            filename = f"fmcg_report_{datetime.now().strftime('%d-%m-%Y %H:%M:%S')}.pdf"
            pdf_bytes = build_pdf_bytes(st.session_state.last_answer, title="FMCG Market Allocation Report")

            with st.chat_message("assistant"):
                st.success(f"PDF sudah siap di-download.")
                st.download_button(
                    label="⬇️ Download PDF",
                    data=pdf_bytes,
                    file_name=filename,
                    mime="application/pdf",
                )
    else:
        # Simpan pesan user
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Bangun prompt yang sudah mengandung info bobot MAS
        mas_instruction = f"""
    Please use the following MAS (Market Allocation Score) weights
    when computing scores for each region:

    - Demand weight w_d = {wd:.3f}
    - Economic weight w_e = {we:.3f}
    - Cost weight w_c = {wc:.3f}

    Use the formula:

    MAS_r = (D_r * w_d) + (E_r * w_e) + (C_r * w_c)

    After computing MAS_r for each region, convert them into allocation proportions:

    Allocation_r = MAS_r / sum_over_regions(MAS)

    Return a markdown table with:
    Region | Demand Factor(D_r) | Economic Factor(E_r) | Cost Factor(C_r) | MAS_r | Allocation_share

    and a short explanation.
    """

        full_user_message = user_input + "\n\n" + mas_instruction

        # Panggil agent
        with st.chat_message("assistant"):
            with st.spinner("Mengumpulkan data makro, menghitung MAS & alokasi..."):
                result = fmcg_supervisor_agent.invoke(
                    {
                        "messages": [
                            {"role": "user", "content": full_user_message}
                        ]
                    },
                    config={
                        "callbacks" : [debug_handler],
                        "tags" : ["fmcg_agent", "streamlit"]
                    },
                    context={"mode": st.session_state["model_mode"]},
                )

            answer = extract_text_from_last_message(result)
            st.markdown(answer)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )
        st.session_state.last_answer = answer


