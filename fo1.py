# app.py
"""
Simple Options Scalper – Streamlit (Signal Generator, No Broker API)

- Logs option buy trades (CE/PE), sets target & stop-loss, and auto-closes on hit.
- Pulls live prices from NSE option chain (public endpoint) – no broker login required.
- Generates **scalping signals** based on short-term momentum of the underlying (index value from NSE),
  picks nearest OTM strike, and proposes entry/target/SL.
- No order placement (per your request). You execute in your broker; app tracks P&L.

Run locally:
    pip install streamlit pandas requests
    streamlit run app.py
"""

import uuid
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import pandas as pd
import streamlit as st
import requests

# -----------------------------
# Config (defaults)
# -----------------------------
DEFAULT_SYMBOL = "MIDCPNIFTY"   # Nifty Midcap Select index options
DEFAULT_LOT = 140               # verify with your broker
STRIKE_STEP = 50                # typical step
DEFAULT_TARGET_ADD = 0.50       # default micro target (₹)
DEFAULT_STOP_SUB = 0.30         # default micro stop  (₹)
DEFAULT_MOM_WINDOW = 6          # last N samples
DEFAULT_MOM_THRESHOLD = 0.15    # % vs MA to bias

# -----------------------------
# Helpers & State
# -----------------------------
@dataclass
class Trade:
    id: str
    ts: str
    symbol: str
    option_type: str  # CE/PE
    strike: float
    expiry: str
    lot_size: int
    lots: int
    entry: float
    target: float
    stop: float
    auto_close: bool
    status: str
    exit_price: float
    pnl: float


def init_state():
    if "trades" not in st.session_state:
        st.session_state.trades: list[dict] = []
    if "default_symbol" not in st.session_state:
        st.session_state.default_symbol = DEFAULT_SYMBOL
    if "default_lot" not in st.session_state:
        st.session_state.default_lot = DEFAULT_LOT
    if "price_hist" not in st.session_state:
        st.session_state.price_hist = []  # list of dicts: {ts, underlying}
    if "last_signal" not in st.session_state:
        st.session_state.last_signal = None


def to_df() -> pd.DataFrame:
    if not st.session_state.trades:
        return pd.DataFrame(columns=[
            "id","ts","symbol","option_type","strike","expiry","lot_size","lots","entry","target","stop","auto_close","status","exit_price","pnl"
        ])
    return pd.DataFrame(st.session_state.trades)


def save_trade(trade: Trade):
    st.session_state.trades.append(asdict(trade))


def update_trade_exit(trade_id: str, exit_price: float):
    for t in st.session_state.trades:
        if t["id"] == trade_id and t["status"] == "OPEN":
            pnl = (exit_price - t["entry"]) * t["lot_size"] * t["lots"]
            t["status"] = "CLOSED"
            t["exit_price"] = float(exit_price)
            t["pnl"] = float(pnl)
            st.toast(f"Closed {t['symbol']} {int(t['strike'])}{t['option_type']} @ {exit_price:.2f} | P&L ₹{pnl:.2f}")
            break

# -----------------------------
# NSE Fetchers (public endpoints)
# -----------------------------
HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br"
}


def fetch_option_chain(symbol: str):
    """Return (records_list, underlyingValue, expiry_list) or (None, None, None)."""
    try:
        session = requests.Session()
        session.get("https://www.nseindia.com", headers=HEADERS, timeout=8)
        url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
        data = session.get(url, headers=HEADERS, timeout=8).json()
        records = data.get("records", {})
        oc = records.get("data", [])
        underlying = records.get("underlyingValue")
        expiries = records.get("expiryDates", [])
        return oc, underlying, expiries
    except Exception as e:
        st.warning(f"NSE fetch failed: {e}")
        return None, None, None


def round_to_step(val: float, step: int) -> int:
    return int(round(val / step) * step)


def nearest_otm_strike(underlying: float, side: str) -> int:
    base = round_to_step(underlying, STRIKE_STEP)
    if side == "CE":
        # OTM Call -> strike just ABOVE underlying
        return base + STRIKE_STEP if base <= underlying else base
    else:
        # OTM Put -> strike just BELOW underlying
        return base - STRIKE_STEP if base >= underlying else base


def option_ltp_from_chain(oc_rows, strike: int, side: str, expiry: str):
    for row in oc_rows:
        if row.get("expiryDate") != expiry:
            continue
        if side == "CE":
            ce = row.get("CE")
            if ce and int(ce.get("strikePrice", 0)) == int(strike):
                return ce.get("lastPrice")
        else:
            pe = row.get("PE")
            if pe and int(pe.get("strikePrice", 0)) == int(strike):
                return pe.get("lastPrice")
    return None

# -----------------------------
# Momentum & Signals
# -----------------------------

def update_price_hist(underlying: float):
    now = datetime.now()
    st.session_state.price_hist.append({"ts": now, "underlying": underlying})
    # keep last 2 hours of data
    cutoff = now - timedelta(hours=2)
    st.session_state.price_hist = [p for p in st.session_state.price_hist if p["ts"] >= cutoff]


def calc_momentum(window: int) -> float:
    """Return % diff of last price vs moving average of last N samples."""
    if window < 1:
        window = 1
    hist = st.session_state.price_hist[-window:]
    if len(hist) < 3:
        return 0.0
    last = hist[-1]["underlying"]
    ma = sum(p["underlying"] for p in hist) / len(hist)
    if ma == 0:
        return 0.0
    return (last - ma) * 100.0 / ma


def generate_signal(symbol: str, expiries: list, oc_rows, underlying: float,
                    target_add: float, stop_sub: float,
                    momentum_threshold: float, window: int):
    if not expiries:
        return None
    # choose nearest expiry
    expiry = expiries[0]

    # momentum-based direction
    mom = calc_momentum(window)
    if mom >= momentum_threshold:
        side = "CE"
    elif mom <= -momentum_threshold:
        side = "PE"
    else:
        # no strong bias; skip signal
        return {
            "status": "HOLD",
            "reason": f"Momentum weak ({mom:.2f}%)",
            "underlying": underlying,
            "ts": datetime.now().strftime("%H:%M:%S")
        }

    strike = nearest_otm_strike(underlying, side)
    ltp = option_ltp_from_chain(oc_rows, strike, side, expiry)
    if ltp is None or ltp <= 0:
        return {
            "status": "HOLD",
            "reason": "No LTP for chosen strike/expiry",
            "underlying": underlying,
            "ts": datetime.now().strftime("%H:%M:%S")
        }

    entry = float(ltp)
    target = round(entry + target_add, 2)
    stop = max(0.0, round(entry - stop_sub, 2))

    sig = {
        "status": "BUY",
        "symbol": symbol,
        "side": side,
        "strike": int(strike),
        "expiry": expiry,
        "entry": round(entry, 2),
        "target": target,
        "stop": stop,
        "underlying": round(underlying, 2),
        "momentum_pct": round(mom, 2),
        "ts": datetime.now().strftime("%H:%M:%S"),
    }
    st.session_state.last_signal = sig
    return sig

# -----------------------------
# Auto-Close engine (checks live LTP and closes at target/SL)
# -----------------------------

def auto_close_if_hit():
    oc_rows, underlying, expiries = fetch_option_chain(st.session_state.default_symbol)
    if oc_rows is None:
        return
    for t in st.session_state.trades:
        if t["status"] != "OPEN" or not t["auto_close"]:
            continue
        ltp = option_ltp_from_chain(oc_rows, int(t["strike"]), t["option_type"], t["expiry"])
        if ltp is None:
            continue
        if ltp >= t["target"]:
            update_trade_exit(t["id"], t["target"])  # book exactly at target
        elif t["stop"] > 0 and ltp <= t["stop"]:
            update_trade_exit(t["id"], t["stop"])    # exit at stop

# -----------------------------
# UI
# -----------------------------
init_state()
st.set_page_config(page_title="Options Scalper (Signals)", layout="wide")
st.title("🟢 Nifty Midcap Options Scalper – Signals (No API)")
st.caption("Generates micro scalping signals and manages manual trades with auto-close.")

with st.sidebar:
    st.subheader("Settings")
    symbol = st.text_input("Symbol", st.session_state.default_symbol, key="symbol_input")

    st.session_state.default_lot = st.number_input("Lot Size", min_value=1, step=1, value=st.session_state.default_lot)
    st.markdown("**Signal Params**")
    mom_win = st.number_input("Momentum window (samples)", min_value=3, value=DEFAULT_MOM_WINDOW)
    mom_thr = st.number_input("Momentum threshold %", min_value=0.05, step=0.05, value=DEFAULT_MOM_THRESHOLD, format="%.2f")
    tgt_add = st.number_input("Target add (₹)", min_value=0.10, step=0.05, value=DEFAULT_TARGET_ADD, format="%.2f")
    sl_sub = st.number_input("Stop subtract (₹)", min_value=0.10, step=0.05, value=DEFAULT_STOP_SUB, format="%.2f")

    st.markdown("---")
    if st.button("Refresh Prices / Generate Signal"):
        oc_rows, underlying, expiries = fetch_option_chain(st.session_state.default_symbol)
        if underlying:
            update_price_hist(underlying)
        sig = generate_signal(
            st.session_state.default_symbol,
            expiries or [], oc_rows or [], underlying or 0,
            tgt_add, sl_sub, mom_thr, int(mom_win)
        )
        auto_close_if_hit()
        if sig and sig.get("status") == "BUY":
            st.success("New BUY signal ready below.")
        elif sig:
            st.info(sig.get("reason", "No signal now."))

# --- Market Snapshot & Momentum
st.header("Market Snapshot & Momentum")
oc_rows, underlying, expiries = fetch_option_chain(st.session_state.default_symbol)
if underlying:
    update_price_hist(underlying)
colA, colB, colC = st.columns(3)
with colA:
    st.metric("Underlying", f"{underlying if underlying else '-'}")
with colB:
    mom = calc_momentum(int(st.session_state.get('mom_win', DEFAULT_MOM_WINDOW)) if 'mom_win' in st.session_state else DEFAULT_MOM_WINDOW)
    # If not stored, recompute with default window for display
    mom = calc_momentum(int(mom_win)) if 'mom_win' in locals() else calc_momentum(DEFAULT_MOM_WINDOW)
    st.metric("Momentum % vs MA", f"{mom:.2f}%")
with colC:
    st.write("Expiry focus:", expiries[0] if expiries else "-")

# --- Signal Panel
st.header("Signal Generator")
if st.session_state.last_signal and st.session_state.last_signal.get("status") == "BUY":
    s = st.session_state.last_signal
    st.success(f"BUY {s['symbol']} {s['strike']}{s['side']} | Exp {s['expiry']} | Entry ₹{s['entry']} | Target ₹{s['target']} | SL ₹{s['stop']} | Mom {s['momentum_pct']}%")
else:
    st.info("Click **Refresh Prices / Generate Signal** in the sidebar to compute a signal.")

# --- New Trade (from signal or manual)
st.header("Enter New Trade")
col1, col2, col3, col4 = st.columns(4)
with col1:
    symbol = st.text_input("Symbol", st.session_state.default_symbol)
with col2:
    option_type = st.selectbox("Type", ["CE", "PE"], index=0)
with col3:
    default_strike = float(st.session_state.last_signal['strike']) if st.session_state.last_signal and st.session_state.last_signal.get('status')=='BUY' else 14000.0
    strike = st.number_input("Strike", min_value=0.0, step=50.0, value=default_strike, format="%.2f")
with col4:
    default_exp = (st.session_state.last_signal['expiry'] if st.session_state.last_signal and st.session_state.last_signal.get('status')=='BUY' else (expiries[0] if expiries else datetime.today().strftime("%d-%b-%Y")))
    expiry = st.text_input("Expiry (as in NSE)", default_exp)

col5, col6, col7, col8 = st.columns(4)
with col5:
    lot_size = st.number_input("Lot Size", min_value=1, step=1,
                           value=st.session_state.default_lot, key="lot_size_main")

with col6:
    lots = st.number_input("Lots", min_value=1, step=1, value=1)
with col7:
    default_entry = float(st.session_state.last_signal['entry']) if st.session_state.last_signal and st.session_state.last_signal.get('status')=='BUY' else 2.40
    entry = st.number_input("Entry (₹)", min_value=0.0, step=0.05, value=default_entry, format="%.2f")
with col8:
    default_target = (float(st.session_state.last_signal['target']) if st.session_state.last_signal and st.session_state.last_signal.get('status')=='BUY' else round(default_entry + DEFAULT_TARGET_ADD, 2))
    target = st.number_input("Target (₹)", min_value=0.0, step=0.05, value=default_target, format="%.2f")

col9, col10 = st.columns(2)
with col9:
    default_stop = (float(st.session_state.last_signal['stop']) if st.session_state.last_signal and st.session_state.last_signal.get('status')=='BUY' else max(0.0, round(default_entry - DEFAULT_STOP_SUB, 2)))
    stop = st.number_input("Stop-Loss (₹)", min_value=0.0, step=0.05, value=default_stop, format="%.2f")
with col10:
    auto_close = st.checkbox("Auto-Close at Target/SL", value=True)

if st.button("➕ Add Trade"):
    t = Trade(
        id=str(uuid.uuid4())[:8],
        ts=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        symbol=symbol.strip().upper(),
        option_type=option_type,
        strike=float(strike),
        expiry=expiry.strip(),
        lot_size=int(lot_size),
        lots=int(lots),
        entry=float(entry),
        target=float(target),
        stop=float(stop),
        auto_close=bool(auto_close),
        status="OPEN",
        exit_price=0.0,
        pnl=0.0,
    )
    save_trade(t)
    st.success(f"Added {t.symbol} {int(t.strike)}{t.option_type} | Entry ₹{t.entry:.2f}, Target ₹{t.target:.2f}, SL ₹{t.stop:.2f}")

# Instant P&L preview
st.markdown(":small_blue_diamond: **Per-lot profit at target** = (Target − Entry) × Lot Size")
per_lot_profit = (target - entry) * lot_size
st.info(f"Per-lot profit at target = ₹{per_lot_profit:.2f}; Total for {lots} lot(s): ₹{per_lot_profit * lots:.2f}")

# --- Blotter & Actions
st.header("Trades Blotter")
df = to_df()
if df.empty:
    st.warning("No trades yet.")
else:
    df = df.sort_values("ts", ascending=False)
    sel_col, act_col1, act_col2 = st.columns([1.2, 0.6, 0.6])
    with sel_col:
        ids = df[df["status"]=="OPEN"]["id"].tolist() or df["id"].tolist()
        pick = st.selectbox("Select Trade ID", ids)
    with act_col1:
        close_px = st.number_input("Manual Close @ (₹)", min_value=0.0, step=0.05, format="%.2f")
        if st.button("Close Selected"):
            update_trade_exit(pick, close_px)
    with act_col2:
        if st.button("Export CSV"):
            file = "scalper_trades.csv"
            df.to_csv(file, index=False)
            st.download_button(label="Download CSV", file_name=file, data=df.to_csv(index=False), mime="text/csv")

    st.dataframe(df, use_container_width=True)

# --- Auto-close button at bottom as well
if st.button("🔁 Check Targets / Stops Now"):
    auto_close_if_hit()
    st.success("Checked open trades against live prices.")
