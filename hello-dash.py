"""Hello-world Streamlit dashboard for XNAT.

Mirrors the auto-install + sys.path pattern from the working
dicom-tag-sniffer dashboard. Use this to confirm XNAT can serve Streamlit
apps before attempting the inference-GUI conversion.

Usage (XNAT):
    Register as a Streamlit dashboard in XNAT admin.
    XNAT sets env vars: XNAT_HOST, XNAT_USER, XNAT_PASS, XNAT_ITEM_ID,
                        XNAT_XSI_TYPE, XNAT_DATA, JUPYTERHUB_ROOT_DIR

Usage (local):
    streamlit run hello-dash.py
"""

import os
import subprocess
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REQUIREMENTS = os.path.join(_SCRIPT_DIR, "requirements-xnat.txt")

if os.path.isfile(_REQUIREMENTS):
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "-r", _REQUIREMENTS],
        stdout=subprocess.DEVNULL,
    )

os.environ.setdefault(
    "STREAMLIT_CONFIG_DIR", os.path.join(_SCRIPT_DIR, ".streamlit")
)

import streamlit as st


_CUSTOM_CSS = """
<style>
    .hello-banner {
        background: linear-gradient(135deg, #1e88e5, #00bcd4);
        color: #fff;
        padding: 18px 24px;
        border-radius: 8px;
        font-size: 18px;
        font-weight: 600;
        margin-bottom: 18px;
    }
    .env-grid code {
        font-size: 12px;
    }
</style>
"""


def main():
    st.set_page_config(
        page_title="Inference Dashboard — Hello",
        page_icon=":hospital:",
        layout="wide",
    )
    st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)

    st.markdown(
        "<div class='hello-banner'>Hello from Streamlit &#x2714;</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "If you can see this, the XNAT Streamlit dashboard runtime is healthy "
        "on this project. Conversion of the inference GUI to Streamlit is the "
        "next step."
    )

    st.divider()
    st.subheader("XNAT environment")

    env_keys = (
        "XNAT_HOST",
        "XNAT_USER",
        "XNAT_ITEM_ID",
        "XNAT_XSI_TYPE",
        "XNAT_DATA",
        "JUPYTERHUB_ROOT_DIR",
    )
    cols = st.columns(2)
    for i, key in enumerate(env_keys):
        val = os.environ.get(key, "<unset>")
        with cols[i % 2]:
            st.markdown(f"**{key}**")
            st.code(val, language=None)

    st.divider()
    st.subheader("Project data probe")

    xnat_item = os.environ.get("XNAT_ITEM_ID", "")
    if xnat_item:
        data_path = os.path.join("/data", "projects", xnat_item)
        if os.path.isdir(data_path):
            try:
                entries = sorted(os.listdir(data_path))[:20]
            except OSError as exc:
                entries = []
                st.warning(f"Could not list `{data_path}`: {exc}")
            st.success(f"Found project mount at `{data_path}`")
            st.write(f"First {len(entries)} entries:")
            st.code("\n".join(entries) or "(empty)", language=None)
        else:
            st.warning(
                f"`{data_path}` does not exist. Project data may not be mounted "
                f"or XNAT_ITEM_ID does not map to a filesystem path on this instance."
            )
    else:
        st.info("XNAT_ITEM_ID is unset — skipping project-data probe.")


if __name__ == "__main__":
    main()
