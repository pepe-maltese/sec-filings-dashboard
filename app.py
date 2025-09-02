"""
BMNR Real-Time SEC Filings Dashboard — Streamlit App

Zero-cost friendly: deploy on Hugging Face Spaces (Streamlit template) or Streamlit Community Cloud.

Requirements (put these in requirements.txt):
  streamlit>=1.37
  requests>=2.32
  beautifulsoup4>=4.12
  lxml>=5.2
  pandas>=2.2
  python-dateutil>=2.9
  openai>=1.40.0

Secrets (Streamlit Cloud → Settings → Secrets):
  SEC_USER_AGENT = "Your Name your@email.com"
  OPENAI_API_KEY = "sk-..."   # optional, for AI summaries
  DEFAULT_CIK = "0001829311"  # optional, BMNR padded to 10 digits
"""

import os
import re
import time
from datetime import datetime
from dateutil import tz

import requests
import pandas as pd
from bs4 import BeautifulSoup

import streamlit as st

# Optional: OpenAI for summaries
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# ------------------------------
# Config
# ------------------------------
def get_secret(name: str, default: str = "") -> str:
    try:
        return st.secrets.get(name, default)
    except Exception:
        return os.getenv(name, default)

SEC_UA = get_secret("SEC_USER_AGENT", "FilingsDashboard/1.0 (contact: please-set-email@example.com)")
DEFAULT_CIK = get_secret("DEFAULT_CIK", "0001829311")

SEC_BASE = "https://data.sec.gov"
ARCHIVES = "https://www.sec.gov/Archives"
HEADERS = {"User-Agent": SEC_UA, "Accept-Encoding": "gzip, deflate"}

st.set_page_config(page_title="SEC Filings Dashboard", page_icon="📄", layout="wide")
st.title("📄 Real-Time SEC Filings Dashboard")
st.caption("Live feed from SEC EDGAR. Uses rule-based summaries, or OpenAI if configured.")

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
    soup = BeautifulSoup(r.text, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text("\n", strip=True)
    return text[:500000]

# ------------------------------
# Rule-based summary
# ------------------------------

KEY_PATTERNS = {
    "financing": r"ATM|at-the-market|equity offering|registered direct|PIPE|warrant|convertible|shelf registration|S-3|ASR|capital raise",
    "insider": r"Form 4|beneficial owner|officer|director|grant|option|restricted stock|RSU",
    "crypto": r"Bitcoin|BTC|Ethereum|ETH|hashrate|miners|mining|immersion|wallet|custody",
    "buyback": r"repurchase|buyback|issuer repurchases|ASC 505-30",
    "guidance": r"outlook|guidance|reaffirm|update|forward-looking",
    "material": r"Item\s*1\.01|Material Definitive Agreement|Item\s*2\.01|acquisition|disposition|Item\s*3\.02|unregistered|Item\s*5\.02|departure|appointment|Item\s*5\.07|shareholder|vote",
}

def generate_rule_based_summary(form: str, text: str) -> dict:
    snippet = text[:4000] if text else ""
    hits = {k: bool(re.search(p, snippet, re.IGNORECASE)) for k, p in KEY_PATTERNS.items()}
    score = 0
    if hits.get("buyback"): score += 2
    if hits.get("financing"): score -= 2
    if hits.get("material"): score += 1
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

# ------------------------------
# OpenAI summary (optional)
# ------------------------------

def ai_summarize(text: str, form: str, model: str = "gpt-4o-mini") -> str:
    key = get_secret("OPENAI_API_KEY")
    if not key or not OpenAI:
        return ""
    try:
        client = OpenAI(api_key=key)
        prompt = f"""You are an equity research assistant. Read the SEC filing excerpt below and write:
- A one-line headline.
- 3–6 bullet points covering material items (financing like ATM/PIPE/warrants, buybacks, guidance, M&A, crypto holdings, and any Item references).
- Keep it factual, concise, and neutral.

Form: {form}

Filing excerpt (may be partial):
{text[:16000]}
"""
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Be terse, precise, and neutral. Avoid speculation."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=600,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        st.info(f"AI summary skipped: {e}")
        return ""

# ------------------------------
# Sidebar
# ------------------------------
with st.sidebar:
    st.header("Settings")
    mode = st.radio("Lookup by", ["Ticker", "CIK"], index=0)
    default_ticker = st.text_input("Ticker (e.g., BMNR)", value="BMNR")
    cik_input = st.text_input("CIK (10 digits)", value=DEFAULT_CIK)
    cik10 = pad_cik(cik_input) if mode == "CIK" else cik_from_ticker(default_ticker)

    st.markdown("---")
    use_ai = st.checkbox("Use OpenAI summaries (optional)", value=False)
    ai_model = st.text_input("OpenAI model", value="gpt-4o-mini") if use_ai else None
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
# Fetch filings
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

cols = [
    "accessionNumber", "filingDate", "reportDate", "acceptanceDateTime",
    "form", "items", "size", "primaryDocument", "primaryDocDescription"
]
rows = []
for i in range(len(recent.get("accessionNumber", []))):
    row = {c: recent.get(c, [None]*len(recent.get("accessionNumber", [])))[i] for c in cols}
    row["url_index"] = f"{ARCHIVES}/edgar/data/{int(cik10)}/{row['accessionNumber'].replace('-', '')}-index.html"
    row["url_primary"] = f"{ARCHIVES}/edgar/data/{int(cik10)}/{row['accessionNumber'].replace('-', '')}/{row['primaryDocument']}"
    rows.append(row)

df = pd.DataFrame(rows)

if forms_filter:
    df = df[df["form"].isin(forms_filter)]
if keyword:
    kw = keyword.strip().lower()
    df = df[df.apply(lambda r: kw in str(r.to_dict()).lower(), axis=1)]

df = df.sort_values("filingDate", ascending=False).head(max_rows).reset_index(drop=True)

# ------------------------------
# Display
# ------------------------------
st.markdown("### Latest Filings")
for idx, r in df.iterrows():
    with st.expander(f"{r['filingDate']} • {r['form']} • {r['primaryDocDescription']}"):
        st.write(f"Accession: {r['accessionNumber']}  ")
        st.write(f"[Index]({r['url_index']}) • [Primary Document]({r['url_primary']})")

        with st.spinner("Downloading & parsing primary document…"):
            try:
                text = get_primary_doc_text(cik10, r['accessionNumber'], r['primaryDocument'])
                time.sleep(0.3)
            except Exception as e:
                st.error(f"Failed to fetch primary document: {e}")
                continue

        summary = generate_rule_based_summary(r['form'], text)
        ai_text = ai_summarize(text, r['form'], ai_model) if use_ai else ""

        impact_color = {"Positive": "#16a34a", "Neutral": "#64748b", "Negative": "#dc2626"}.get(summary["impact"], "#64748b")
        st.markdown(f"""
            <div style='display:inline-block;padding:4px 10px;border-radius:12px;background:{impact_color};color:#fff;font-weight:600;'>
                {summary['impact']}
            </div>
        """, unsafe_allow_html=True)

        if ai_text:
            st.markdown("**OpenAI Summary:**")
            st.write(ai_text)
        else:
            st.markdown(f"**Headline:** {summary['headline']}")
            if summary["bullets"]:
                st.markdown("**Signals detected:**")
                for b in summary["bullets"]:
                    st.markdown(f"- {b}")

        st.markdown("**Document preview (first ~8k chars):**")
        st.code(text[:8000])

st.caption("Data: SEC EDGAR. Summaries are heuristic or AI-generated — not investment advice.")
