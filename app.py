"""
Real-Time SEC Filings Dashboard — Streamlit app

- Rule-based summaries (free) + optional OpenAI single-paragraph summaries
- Throttled requests to avoid 429/rate-limits
- Cached AI summaries per filing to avoid re-paying
- No document preview; shows headline → signals → one-paragraph summary
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
# Basic config & helpers
# ------------------------------
def get_secret(name: str, default: str = "") -> str:
    """Read from st.secrets or environment."""
    try:
        return st.secrets.get(name, default)
    except Exception:
        return os.getenv(name, default)

SEC_UA = get_secret("SEC_USER_AGENT", "FilingsDashboard/1.0 (contact: please-set-email@example.com)")
DEFAULT_CIK = get_secret("DEFAULT_CIK", "0001829311")

SEC_BASE = "https://data.sec.gov"
ARCHIVES = "https://www.sec.gov/Archives"
HEADERS = {"User-Agent": SEC_UA, "Accept-Encoding": "gzip, deflate"}

# Streamlit page
st.set_page_config(page_title="SEC Filings Dashboard", page_icon="📄", layout="wide")
st.title("📄 Real-Time SEC Filings Dashboard")
st.caption("Free rule-based summaries. Optional OpenAI one-paragraph summaries (with throttling & caching).")

# Respect read-only roots (harmless elsewhere)
os.environ.setdefault("STREAMLIT_HOME", "/tmp")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")

def pad_cik(cik: str) -> str:
    s = re.sub(r"\D", "", cik or "")
    return s.zfill(10) if s else ""

# ------------------------------
# HTTP helpers (polite + retry)
# ------------------------------
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(HEADERS)
    retry = Retry(
        total=4, read=4, connect=4,
        backoff_factor=0.6,
        status_forcelist=[403, 429, 500, 502, 503, 504],
        allowed_methods=["GET"], raise_on_status=False
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("https://", adapter); s.mount("http://", adapter)
    return s

SESSION = make_session()

@st.cache_data(show_spinner=False, ttl=900)
def fetch_company_submissions(cik10: str) -> dict:
    url = f"{SEC_BASE}/submissions/CIK{cik10}.json"
    r = SESSION.get(url, timeout=30)
    if r.status_code == 403:
        st.warning("SEC returned 403 (Forbidden). Check your SEC_USER_AGENT in Settings → Secrets.")
    r.raise_for_status()
    return r.json()

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_company_ticker_map() -> pd.DataFrame:
    url = f"{SEC_BASE}/files/company_tickers.json"
    r = SESSION.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()
    rows = [{"cik": str(x.get("cik_str","")).zfill(10), "ticker": x.get("ticker",""), "title": x.get("title","")} for x in data]
    return pd.DataFrame(rows)

@st.cache_data(show_spinner=False, ttl=3600)
def cik_from_ticker(ticker: str) -> str:
    try:
        if not ticker: return ""
        df = fetch_company_ticker_map()
        hit = df[df["ticker"].str.upper() == ticker.upper()]
        return pad_cik(hit.iloc[0]["cik"]) if not hit.empty else ""
    except Exception:
        st.warning("Could not resolve ticker (rate-limit or not found). Try CIK mode.")
        return ""

@st.cache_data(show_spinner=False, ttl=3600)
def get_primary_doc_text(cik10: str, accession_no: str, primary_doc: str) -> str:
    acc_nodash = accession_no.replace("-", "")
    url = f"{ARCHIVES}/edgar/data/{int(cik10)}/{acc_nodash}/{primary_doc}"
    r = SESSION.get(url, timeout=60)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")
    for tag in soup(["script", "style"]): tag.decompose()
    text = soup.get_text("\n", strip=True)
    return text[:500000]

# ------------------------------
# Rule-based signals (free)
# ------------------------------
KEY_PATTERNS = {
    "financing": r"ATM|at-the-market|equity offering|registered direct|PIPE|warrant|convertible|shelf registration|S-3|ASR|capital raise",
    "insider": r"Form 4|beneficial owner|officer|director|grant|option|restricted stock|RSU",
    "crypto": r"Bitcoin|BTC|Ethereum|ETH|hashrate|miners|mining|immersion|wallet|custody",
    "buyback": r"repurchase|buyback|issuer repurchases|ASC 505-30",
    "guidance": r"outlook|guidance|reaffirm|update|forward-looking",
    "material": r"Item\s*1\.01|Material Definitive Agreement|Item\s*2\.01|acquisition|disposition|Item\s*3\.02|unregistered|Item\s*5\.02|departure|appointment|Item\s*5\.07|shareholder|vote",
}

def rule_summary(form: str, text: str) -> dict:
    snippet = text[:4000] if text else ""
    hits = {k: bool(re.search(p, snippet, re.IGNORECASE)) for k, p in KEY_PATTERNS.items()}
    score = 0
    if hits.get("buyback"): score += 2
    if hits.get("financing"): score -= 2
    if hits.get("material"): score += 1
    impact = "Positive" if score >= 2 else ("Negative" if score <= -2 else "Neutral")
    headline = f"{form}: {impact} — " + (
        "buyback mentioned" if hits.get("buyback") else
        "financing/dilution signals" if hits.get("financing") else
        "material agreement or event" if hits.get("material") else
        "insider/ownership update" if hits.get("insider") else
        "no strong signal"
    )
    bullets = []
    if hits.get("material"): bullets.append("Material item(s) indicated (e.g., Item 1.01/2.01/5.02/5.07).")
    if hits.get("financing"): bullets.append("Financing activity detected (ATM/PIPE/warrants/shelf). Potential dilution risk.")
    if hits.get("buyback"): bullets.append("Repurchase/buyback language detected.")
    if hits.get("insider"): bullets.append("Insider/beneficial ownership or equity grants referenced.")
    if hits.get("crypto"): bullets.append("Crypto/mining references present (BTC/ETH/hashrate).")
    if hits.get("guidance"): bullets.append("Guidance/outlook language present.")
    return {"impact": impact, "headline": headline, "bullets": bullets, "flags": hits}

def compact_paragraph_from_rule(rs: dict) -> str:
    """Turn rule-based headline + bullets into a single paragraph."""
    bullets_text = " ".join(b.rstrip(".") + "." for b in rs.get("bullets", []))
    return (rs["headline"] + (" " + bullets_text if bullets_text else "")).strip()

# ------------------------------
# OpenAI single-paragraph summary (optional) + caching
# ------------------------------
@st.cache_data(show_spinner=False, ttl=7*24*3600)  # cache for a week
def cached_ai_summary(accession: str, form: str, text_excerpt: str, model: str, key: str) -> str:
    if not key or not OpenAI:
        return ""
    client = OpenAI(api_key=key)
    prompt = f"""Summarize the SEC filing excerpt in ONE concise paragraph (4–6 sentences).
Focus on financing (ATM/PIPE/warrants), buybacks, guidance, M&A, crypto holdings, and any Item references (1.01/2.01/3.02/5.02/5.07).
Be factual, neutral, and precise. Avoid speculation.

Form: {form}

Filing excerpt (may be partial):
{text_excerpt}
"""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a precise equity research assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=300,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return ""  # fallback handled by caller

def ai_summarize_one_paragraph(accession: str, form: str, text: str, model: str, delay_s: float) -> str:
    key = get_secret("OPENAI_API_KEY", "")
    if not key or not OpenAI:
        return ""
    time.sleep(max(0.0, delay_s))  # pacing before each OpenAI call
    excerpt = (text or "")[:6000]  # keep tokens in check
    return cached_ai_summary(accession, form, excerpt, model, key)

# ------------------------------
# Sidebar controls
# ------------------------------
with st.sidebar:
    st.header("Settings")
    # Default to CIK to avoid ticker-map calls on first render
    mode = st.radio("Lookup by", ["Ticker", "CIK"], index=1)
    if mode == "Ticker":
        default_ticker = st.text_input("Ticker (e.g., BMNR)", value="BMNR")
        cik10 = cik_from_ticker(default_ticker)
    else:
        cik_input = st.text_input("CIK (10 digits)", value=DEFAULT_CIK)
        cik10 = pad_cik(cik_input)

    st.markdown("---")
    use_ai = st.checkbox("Use OpenAI one-paragraph summaries (optional)", value=False)
    ai_model = st.text_input("OpenAI model", value="gpt-4o-mini") if use_ai else None
    max_ai = st.slider("Max AI summaries this run", 0, 20, 5, help="Past this, rule-based summaries are shown.")
    ai_delay = st.slider("Delay before each OpenAI call (sec)", 0.0, 3.0, 1.2, 0.1)
    sec_delay = st.slider("Delay between SEC doc downloads (sec)", 0.0, 3.0, 1.0, 0.1)

    st.markdown("---")
    max_rows = st.slider("Number of recent filings", 1, 50, 5, step=1, help="Always shows the most recent filings first")
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
# Fetch company filings
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
# Render filings (no preview; pacing + caching)
# ------------------------------
st.markdown("### Latest Filings")

ai_calls = 0
for _, r in df.iterrows():
    with st.expander(f"{r['filingDate']} • {r['form']} • {r['primaryDocDescription']}"):
        st.write(f"Accession: {r['accessionNumber']}")
        st.write(f"[Index]({r['url_index']}) • [Primary Document]({r['url_primary']})")

        with st.spinner("Downloading & parsing primary document…"):
            try:
                time.sleep(sec_delay)  # polite pause before each SEC doc fetch
                text = get_primary_doc_text(cik10, r['accessionNumber'], r['primaryDocument'])
            except Exception as e:
                st.error(f"Failed to fetch primary document: {e}")
                continue

        # Rule-based summary (always available)
        rs = rule_summary(r['form'], text)

        # Optional AI one-paragraph summary (throttled & cached)
        ai_text = ""
        if use_ai and ai_calls < max_ai:
            ai_text = ai_summarize_one_paragraph(r['accessionNumber'], r['form'], text, ai_model, ai_delay)
            if ai_text:
                ai_calls += 1

        # Impact pill
        impact_color = {"Positive": "#16a34a", "Neutral": "#64748b", "Negative": "#dc2626"}.get(rs["impact"], "#64748b")
        st.markdown(
            f"<div style='display:inline-block;padding:4px 10px;border-radius:12px;background:{impact_color};color:#fff;font-weight:600;'>{rs['impact']}</div>",
            unsafe_allow_html=True
        )

        # Headline → signals → one-paragraph summary
        st.markdown(f"**Rule-based headline:** {rs['headline']}")
        if rs["bullets"]:
            st.markdown("**Signals detected:**")
            for b in rs["bullets"]:
                st.markdown(f"- {b}")

        st.markdown("**Summary (one paragraph):**")
        if ai_text:
            st.write(ai_text)
        else:
            st.write(compact_paragraph_from_rule(rs))

st.caption("Data: SEC EDGAR. Summaries are heuristic or AI-generated — not investment advice.")
