"""
BMNR Real‑Time SEC Filings Dashboard — Streamlit App

Zero‑cost friendly: deploy on Hugging Face Spaces (Streamlit template) or Streamlit Community Cloud.

Requirements (put these in requirements.txt if you split files):
  streamlit>=1.37
  requests>=2.32
  beautifulsoup4>=4.12
  lxml>=5.2
  pandas>=2.2
  python-dateutil>=2.9

Optional env vars (set in your Space / Streamlit Cloud):
  SEC_USER_AGENT="Your Name your.email@example.com"   # SEC requires a descriptive UA
  DEFAULT_CIK="0001829311"  # BMNR padded to 10 digits (changeable in UI)

Notes:
- SEC asks clients to include a descriptive User‑Agent and to be respectful with rate limits (<=10 req/sec). We also add short sleeps and caching.
- This app uses SEC's submissions JSON for speed, fetches specific filing documents on demand, and generates light rule‑based summaries locally (no paid AI required).
- You can track any ticker/CIK, not just BMNR.
"""

import os
import time
import re
from datetime import datetime
from dateutil import tz

import requests
import pandas as pd
from bs4 import BeautifulSoup

import streamlit as st

# ------------------------------
# Config
# ------------------------------
SEC_UA = os.getenv("SEC_USER_AGENT", "FilingsDashboard/1.0 (contact: please-set-email@example.com)")
DEFAULT_CIK = os.getenv("DEFAULT_CIK", "0001829311")  # BMNR
SEC_BASE = "https://data.sec.gov"
ARCHIVES = "https://www.sec.gov/Archives"

HEADERS = {"User-Agent": SEC_UA, "Accept-Encoding": "gzip, deflate"}

st.set_page_config(page_title="SEC Filings Dashboard", page_icon="📄", layout="wide")
st.title("📄 Real‑Time SEC Filings Dashboard")
st.caption("Zero‑cost deployable. Live feed from SEC EDGAR with local, rule‑based summaries.")

# ------------------------------
# Helpers
# ------------------------------

def pad_cik(cik: str) -> str:
    s = re.sub(r"\D", "", cik or "")
    return s.zfill(10) if s else ""

@st.cache_data(show_spinner=False, ttl=900)
def fetch_company_submissions(cik10: str) -> dict:
    url = f"{SEC_BASE}/submissions/CIK{cik10}.json"
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_company_ticker_map() -> pd.DataFrame:
    url = f"{SEC_BASE}/files/company_tickers.json"
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    data = r.json()
    # Convert to DataFrame
    rows = []
    for item in data:
        rows.append({
            "cik": str(item.get("cik_str", "")).zfill(10),
            "ticker": item.get("ticker", ""),
            "title": item.get("title", ""),
        })
    return pd.DataFrame(rows)

@st.cache_data(show_spinner=False, ttl=3600)
def cik_from_ticker(ticker: str) -> str:
    if not ticker:
        return ""
    df = fetch_company_ticker_map()
    hit = df[df["ticker"].str.upper() == ticker.upper()]
    if hit.empty:
        return ""
    return pad_cik(hit.iloc[0]["cik"])

@st.cache_data(show_spinner=False, ttl=3600)
def get_primary_doc_text(cik10: str, accession_no: str, primary_doc: str) -> str:
    acc_nodash = accession_no.replace("-", "")
    url = f"{ARCHIVES}/edgar/data/{int(cik10)}/{acc_nodash}/{primary_doc}"
    r = requests.get(url, headers=HEADERS, timeout=60)
    r.raise_for_status()
    # Parse HTML -> text
    soup = BeautifulSoup(r.text, "lxml")
    # Remove scripts/styles
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text("\n", strip=True)
    # Trim overly long text to keep UI responsive
    return text[:500000]

KEY_PATTERNS = {
    "financing": r"ATM|at-the-market|equity offering|registered direct|PIPE|warrant|convertible|shelf registration|S-3|ASR|capital raise",
    "insider": r"Form 4|beneficial owner|officer|director|grant|option|restricted stock|RSU",
    "crypto": r"Bitcoin|BTC|Ethereum|ETH|hashrate|miners|mining|immersion|wallet|custody",
    "buyback": r"repurchase|buyback|issuer repurchases|ASC 505-30",
    "guidance": r"outlook|guidance|reaffirm|update|forward-looking",
    "material": r"Item\s*1\.01|Material Definitive Agreement|Item\s*2\.01|acquisition|disposition|Item\s*3\.02|unregistered|Item\s*5\.02|departure|appointment|Item\s*5\.07|shareholder|vote",
}

TAG_WEIGHTS = {
    "Positive": ["buyback",],
    "Negative": ["financing"],
    "Neutral": ["insider", "guidance"],
}

def generate_rule_based_summary(form: str, text: str) -> dict:
    snippet = text[:4000] if text else ""
    hits = {k: bool(re.search(p, snippet, re.IGNORECASE)) for k, p in KEY_PATTERNS.items()}
    # Simple impact scoring
    score = 0
    if hits.get("buyback"): score += 2
    if hits.get("financing"): score -= 2
    if hits.get("material"): score += 1
    if hits.get("insider"): score -= 0  # neutral
    if hits.get("crypto"): score += 0   # informational
    impact = "Positive" if score >= 2 else ("Negative" if score <= -2 else "Neutral")

    bullets = []
    if hits.get("material"): bullets.append("Material item(s) indicated (e.g., Item 1.01/2.01/5.02/5.07).")
    if hits.get("financing"): bullets.append("Financing activity detected (ATM/PIPE/warrants/shelf). Potential dilution risk.")
    if hits.get("buyback"): bullets.append("Repurchase/buyback language detected.")
    if hits.get("insider"): bullets.append("Insider/beneficial ownership or equity grants referenced.")
    if hits.get("crypto"): bullets.append("Crypto/mining references present (BTC/ETH/hashrate).")
    if hits.get("guidance"): bullets.append("Guidance/outlook language present.")

    headline = f"{form}: {impact} — "
    if hits.get("buyback"): headline += "buyback mentioned"
    elif hits.get("financing"): headline += "financing/dilution signals"
    elif hits.get("material"): headline += "material agreement or event"
    elif hits.get("insider"): headline += "insider/ownership update"
    else: headline += "no strong signal"

    return {"impact": impact, "headline": headline, "bullets": bullets, "flags": hits}


def to_local(date_str: str) -> str:
    try:
        # SEC uses YYYY-MM-DD
        dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=tz.tzutc())
        local = dt.astimezone(tz.tzlocal())
        return local.strftime("%Y-%m-%d")
    except Exception:
        return date_str

# ------------------------------
# Sidebar Inputs
# ------------------------------
with st.sidebar:
    st.header("Settings")
    mode = st.radio("Lookup by", ["Ticker", "CIK"], index=0)
    default_ticker = st.text_input("Ticker (e.g., BMNR)", value="BMNR")
    cik_input = st.text_input("CIK (10 digits)", value=DEFAULT_CIK)
    cik10 = pad_cik(cik_input) if mode == "CIK" else cik_from_ticker(default_ticker)

    st.markdown("---")
    max_rows = st.slider("Max filings to show", 5, 100, 30, step=5)
    forms_filter = st.multiselect(
        "Filter form types",
        ["8-K", "10-Q", "10-K", "S-3", "S-3ASR", "S-1", "4", "13D", "13G", "SC 13D", "SC 13G"],
        default=[]
    )
    keyword = st.text_input("Keyword filter (optional)", placeholder="ATM, buyback, warrant, Ethereum…")

if not cik10:
    st.info("Enter a ticker or CIK to begin.")
    st.stop()

# ------------------------------
# Fetch submissions
# ------------------------------
with st.spinner("Fetching company submissions from SEC…"):
    try:
        company = fetch_company_submissions(cik10)
    except Exception as e:
        st.error(f"Failed to fetch submissions: {e}")
        st.stop()

name = company.get("name", "")
st.subheader(f"{name} — CIK {cik10}")

recent = company.get("filings", {}).get("recent", {})
if not recent:
    st.warning("No recent filings found.")
    st.stop()

# Build DataFrame of recent filings
cols = [
    "accessionNumber", "filingDate", "reportDate", "acceptanceDateTime",
    "act", "form", "fileNumber", "filmNumber", "items", "size",
    "isInlineXBRL", "isXBRL", "primaryDocument", "primaryDocDescription"
]
rows = []
for i in range(len(recent.get("accessionNumber", []))):
    row = {c: recent.get(c, [None]*len(recent.get("accessionNumber", [])))[i] for c in cols}
    row["url_index"] = f"{ARCHIVES}/edgar/data/{int(cik10)}/{row['accessionNumber'].replace('-', '')}-index.html"
    row["url_primary"] = f"{ARCHIVES}/edgar/data/{int(cik10)}/{row['accessionNumber'].replace('-', '')}/{row['primaryDocument']}"
    rows.append(row)

df = pd.DataFrame(rows)

# Apply filters
if forms_filter:
    df = df[df["form"].isin(forms_filter)]
if keyword:
    kw = keyword.strip().lower()
    df = df[df.apply(lambda r: kw in str(r.to_dict()).lower(), axis=1)]

df = df.sort_values("filingDate", ascending=False).head(max_rows).reset_index(drop=True)

# ------------------------------
# Display Table
# ------------------------------
st.markdown("### Latest Filings")

# Pretty table
display_df = df[["filingDate", "form", "primaryDocDescription", "accessionNumber", "url_index", "url_primary"]].copy()
display_df.rename(columns={
    "filingDate": "Date",
    "form": "Form",
    "primaryDocDescription": "Description",
    "accessionNumber": "Accession",
    "url_index": "Index",
    "url_primary": "Primary Doc",
}, inplace=True)

# Convert to clickable links in Streamlit using unsafe_allow_html for this table
def linkify(url, text):
    return f"<a href='{url}' target='_blank'>{text}</a>"

display_html = "<table>\n<tr><th>Date</th><th>Form</th><th>Description</th><th>Accession</th><th>Links</th></tr>"
for _, r in display_df.iterrows():
    links = f"{linkify(r['Index'], 'Index')} | {linkify(r['Primary Doc'], 'Doc')}"
    display_html += f"<tr>" \
                    f"<td>{r['Date']}</td>" \
                    f"<td>{r['Form']}</td>" \
                    f"<td>{(r['Description'] or '')[:120]}</td>" \
                    f"<td>{r['Accession']}</td>" \
                    f"<td>{links}</td>" \
                    f"</tr>"
display_html += "</table>"

st.markdown(display_html, unsafe_allow_html=True)

st.markdown("---")

# ------------------------------
# Per‑filing Summaries
# ------------------------------
for idx, r in df.iterrows():
    with st.expander(f"{r['filingDate']} • {r['form']} • {r['primaryDocDescription']}"):
        st.write(f"Accession: {r['accessionNumber']}  ")
        st.write(f"[Index]({r['url_index']}) • [Primary Document]({r['url_primary']})")

        with st.spinner("Downloading & parsing primary document…"):
            try:
                text = get_primary_doc_text(cik10, r['accessionNumber'], r['primaryDocument'])
                # light throttle to be extra kind to SEC
                time.sleep(0.3)
            except Exception as e:
                st.error(f"Failed to fetch primary document: {e}")
                continue

        summary = generate_rule_based_summary(r['form'], text)

        # Render impact pill
        impact_color = {"Positive": "#16a34a", "Neutral": "#64748b", "Negative": "#dc2626"}.get(summary["impact"], "#64748b")
        st.markdown(f"""
            <div style='display:inline-block;padding:4px 10px;border-radius:12px;background:{impact_color};color:#fff;font-weight:600;'>
                {summary['impact']}
            </div>
        """, unsafe_allow_html=True)

        st.markdown(f"**Headline:** {summary['headline']}")
        if summary["bullets"]:
            st.markdown("**Signals detected:**")
            for b in summary["bullets"]:
                st.markdown(f"- {b}")

        # Show preview of the filing text with keyword highlighting
        preview = text[:8000]
        # rudimentary highlighting
        if keyword:
            patt = re.compile(re.escape(keyword), re.IGNORECASE)
            preview = patt.sub(lambda m: f"**{m.group(0)}**", preview)
        st.markdown("**Document preview (first ~8k chars):**")
        st.code(preview)

st.markdown("---")
st.caption("Data: SEC EDGAR. Summaries are heuristic and informational — not investment advice.")
