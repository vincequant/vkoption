import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import requests
import streamlit as st

OPTION_CONTRACT_SIZE = 100
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
STORAGE_FILE = DATA_DIR / "options_portfolio.json"
POSITION_TYPE_CHOICES = ["short_put", "short_call", "stock_long", "stock_sell"]


def normalize_symbol(symbol: str) -> str:
    return str(symbol or "").strip().upper()


def infer_market(symbol: str) -> str:
    normalized = normalize_symbol(symbol)
    if normalized.endswith(".HK") or normalized.isdigit() and len(normalized) <= 5:
        return "hk"
    if normalized.startswith("SH") or normalized.startswith("SZ"):
        return "cn"
    if normalized.isdigit() and len(normalized) == 6:
        return "cn"
    return "us"


def market_currency_symbol(market: str) -> str:
    if market == "hk":
        return "HK$"
    if market == "cn":
        return "¥"
    return "$"


def position_type_label(position_type: str) -> str:
    labels = {
        "short_put": "Short Put (卖出看跌)",
        "short_call": "Short Call (卖出看涨)",
        "stock_long": "正股（买入）",
        "stock_sell": "正股（卖出）",
    }
    return labels.get(position_type, "Short Put (卖出看跌)")


def position_multiplier(position_type: str) -> int:
    return OPTION_CONTRACT_SIZE if position_type in {"short_put", "short_call"} else 1


def position_direction(position_type: str) -> int:
    return -1 if position_type == "stock_sell" else 1


def fmp_symbol_candidates(symbol: str, market: str) -> List[str]:
    normalized = normalize_symbol(symbol)
    candidates: List[str] = [normalized]

    if market == "hk":
        raw = normalized.replace(".HK", "")
        if raw.isdigit():
            padded = raw.zfill(5)
            candidates.extend([f"{raw}.HK", f"{padded}.HK", raw, padded])

    if market == "cn":
        raw = normalized
        if raw.startswith("SH") and raw[2:].isdigit():
            candidates.extend([f"{raw[2:]}.SS", raw[2:]])
        elif raw.startswith("SZ") and raw[2:].isdigit():
            candidates.extend([f"{raw[2:]}.SZ", raw[2:]])
        elif raw.isdigit() and len(raw) == 6:
            suffix = ".SS" if raw.startswith("6") else ".SZ"
            candidates.extend([f"{raw}{suffix}", raw])

    ordered = []
    seen = set()
    for item in candidates:
        key = normalize_symbol(item)
        if key and key not in seen:
            seen.add(key)
            ordered.append(key)
    return ordered


def ensure_state() -> None:
    if "options_holdings" not in st.session_state:
        st.session_state.options_holdings = []
    if "last_updated" not in st.session_state:
        st.session_state.last_updated = None
    if "last_saved" not in st.session_state:
        st.session_state.last_saved = None
    if "holdings_loaded" not in st.session_state:
        st.session_state.holdings_loaded = False
    if not st.session_state.holdings_loaded:
        loaded = load_holdings_from_disk()
        st.session_state.options_holdings = loaded
        st.session_state.holdings_loaded = True


def load_holdings_from_disk() -> List[Dict]:
    try:
        if not STORAGE_FILE.exists():
            return []
        payload = json.loads(STORAGE_FILE.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return []
        items = payload.get("holdings", [])
        last_updated = payload.get("last_updated")
        last_saved = payload.get("last_saved")
        if isinstance(last_updated, str):
            st.session_state.last_updated = last_updated
        if isinstance(last_saved, str):
            st.session_state.last_saved = last_saved
        if isinstance(items, list):
            cleaned = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                position_type = str(item.get("option_type") or item.get("position_type") or "short_put")
                if position_type == "stock_from_put":
                    position_type = "stock_long"
                if position_type not in POSITION_TYPE_CHOICES:
                    position_type = "short_put"
                cleaned.append(
                    {
                        "id": item.get("id"),
                        "symbol": normalize_symbol(item.get("symbol", "")),
                        "market": item.get("market") or infer_market(item.get("symbol", "")),
                        "quantity": int(item.get("quantity", 1) or 1),
                        "strike_price": float(item.get("strike_price", 0) or 0),
                        "position_type": position_type,
                        "current_price": item.get("current_price"),
                        "created_at": item.get("created_at"),
                    }
                )
            return cleaned
        return []
    except (json.JSONDecodeError, OSError):
        return []


def save_holdings_to_disk() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    payload = {
        "last_updated": st.session_state.last_updated,
        "last_saved": now_str,
        "holdings": st.session_state.options_holdings,
    }
    STORAGE_FILE.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    st.session_state.last_saved = now_str


def fetch_prices_from_fmp(items: List[Dict], api_key: str) -> Dict[str, float]:
    if not api_key:
        return {}

    item_candidates: Dict[str, List[str]] = {}
    query_symbols: List[str] = []
    for item in items:
        symbol = normalize_symbol(item["symbol"])
        market = item.get("market") or infer_market(symbol)
        key = f"{market}:{symbol}"
        candidates = fmp_symbol_candidates(symbol, market)
        item_candidates[key] = candidates
        query_symbols.extend(candidates)

    query_symbols = list(dict.fromkeys(query_symbols))
    quote_map: Dict[str, float] = {}

    for i in range(0, len(query_symbols), 100):
        chunk = query_symbols[i : i + 100]
        symbols_str = ",".join(chunk)
        url = f"https://financialmodelingprep.com/api/v3/quote/{symbols_str}"
        try:
            resp = requests.get(url, params={"apikey": api_key}, timeout=10)
            if resp.status_code != 200:
                continue
            payload = resp.json()
            if not isinstance(payload, list):
                continue
            for quote in payload:
                if not isinstance(quote, dict):
                    continue
                sym = normalize_symbol(quote.get("symbol", ""))
                price = quote.get("price")
                if sym and isinstance(price, (int, float)):
                    quote_map[sym] = float(price)
                    quote_map[sym.replace(".HK", "")] = float(price)
                    quote_map[sym.lstrip("0")] = float(price)
        except requests.RequestException:
            continue

    resolved: Dict[str, float] = {}
    for key, candidates in item_candidates.items():
        matched = None
        for candidate in candidates:
            if candidate in quote_map:
                matched = quote_map[candidate]
                break
            if candidate.replace(".HK", "") in quote_map:
                matched = quote_map[candidate.replace(".HK", "")]
                break
            if candidate.lstrip("0") in quote_map:
                matched = quote_map[candidate.lstrip("0")]
                break
        if matched is not None:
            resolved[key] = matched

    return resolved


def fetch_single_price(symbol: str, market: str, api_key: str) -> float | None:
    if not api_key:
        return None
    candidates = fmp_symbol_candidates(symbol, market)
    if not candidates:
        return None
    url = f"https://financialmodelingprep.com/api/v3/quote/{','.join(candidates)}"
    try:
        resp = requests.get(url, params={"apikey": api_key}, timeout=10)
        if resp.status_code != 200:
            return None
        payload = resp.json()
        if not isinstance(payload, list):
            return None
        quote_map: Dict[str, float] = {}
        for quote in payload:
            if not isinstance(quote, dict):
                continue
            sym = normalize_symbol(quote.get("symbol", ""))
            price = quote.get("price")
            if sym and isinstance(price, (int, float)):
                quote_map[sym] = float(price)
                quote_map[sym.replace(".HK", "")] = float(price)
                quote_map[sym.lstrip("0")] = float(price)
        for c in candidates:
            if c in quote_map:
                return quote_map[c]
            if c.replace(".HK", "") in quote_map:
                return quote_map[c.replace(".HK", "")]
            if c.lstrip("0") in quote_map:
                return quote_map[c.lstrip("0")]
        return None
    except requests.RequestException:
        return None


def portfolio_summary(holdings: List[Dict]) -> Dict[str, float]:
    total_value = 0.0
    assignment_value = 0.0
    for item in holdings:
        qty = float(item.get("quantity", 0) or 0)
        strike = float(item.get("strike_price", 0) or 0)
        position_type = item.get("position_type", "short_put")
        multiplier = position_multiplier(position_type)
        direction = position_direction(position_type)
        current = item.get("current_price")
        if isinstance(current, (int, float)):
            total_value += float(current) * qty * multiplier * direction
        if position_type == "short_put":
            assignment_value += strike * qty * OPTION_CONTRACT_SIZE
    return {
        "total_value": total_value,
        "assignment_value": assignment_value,
        "count": len(holdings),
    }


def add_holding(symbol: str, quantity: int, strike_price: float, position_type: str) -> None:
    normalized = normalize_symbol(symbol)
    st.session_state.options_holdings.append(
        {
            "id": f"{int(datetime.now().timestamp() * 1000)}_{os.urandom(3).hex()}",
            "symbol": normalized,
            "market": infer_market(normalized),
            "quantity": int(quantity),
            "strike_price": float(strike_price),
            "position_type": position_type,
            "current_price": None,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
    )
    save_holdings_to_disk()


def update_prices() -> None:
    holdings = st.session_state.options_holdings
    if not holdings:
        st.warning("当前没有持仓。")
        return

    api_key = st.session_state.get("fmp_api_key", "").strip()
    if not api_key:
        st.error("请先在侧边栏填写 FMP API Key。")
        return

    price_map = fetch_prices_from_fmp(holdings, api_key)
    updated = 0
    for item in holdings:
        key = f"{item['market']}:{normalize_symbol(item['symbol'])}"
        if key in price_map:
            item["current_price"] = float(price_map[key])
            updated += 1

    st.session_state.last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    save_holdings_to_disk()
    if updated == 0:
        st.warning("未获取到可匹配报价，请检查标的代码格式。")
    else:
        st.success(f"更新完成：{updated}/{len(holdings)} 条持仓已刷新。")


def remove_holding(holding_id: str) -> None:
    st.session_state.options_holdings = [
        item for item in st.session_state.options_holdings if item["id"] != holding_id
    ]
    save_holdings_to_disk()


st.markdown(
    """
<style>
    .block-container {
        max-width: 430px;
        padding-top: 1rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3 {
        letter-spacing: 0.2px;
    }
    .metric-card {
        background: linear-gradient(180deg, #ffffff 0%, #f6f8fc 100%);
        border: 1px solid #e8edf5;
        border-radius: 14px;
        padding: 14px 16px;
        margin-bottom: 10px;
    }
    .metric-label {
        color: #64748b;
        font-size: 0.85rem;
        margin-bottom: 6px;
    }
    .metric-value {
        color: #1f2937;
        font-size: 1.65rem;
        font-weight: 700;
        line-height: 1.1;
        word-break: break-word;
    }
    .pill-risk {
        display: inline-block;
        border-radius: 999px;
        padding: 3px 10px;
        font-size: 0.78rem;
        font-weight: 600;
        color: #b42318;
        background: #fee4e2;
        border: 1px solid #fecdca;
    }
    .pill-ok {
        display: inline-block;
        border-radius: 999px;
        padding: 3px 10px;
        font-size: 0.78rem;
        font-weight: 600;
        color: #027a48;
        background: #ecfdf3;
        border: 1px solid #abefc6;
    }
    .stButton > button, .stDownloadButton > button {
        width: 100%;
        min-height: 48px;
        font-weight: 700;
        border-radius: 12px;
    }
</style>
""",
    unsafe_allow_html=True,
)

ensure_state()

st.title("期权 + 正股组合")
last_updated = st.session_state.last_updated or "--"
st.caption(f"最后更新：{last_updated}")
st.caption(f"本地保存：{st.session_state.last_saved or '--'}")

with st.sidebar:
    st.subheader("数据源设置")
    default_key = st.secrets.get("FMP_API_KEY", os.getenv("FMP_API_KEY", ""))
    st.session_state.fmp_api_key = st.text_input(
        "FMP API Key", value=default_key, type="password", help="优先读取 secrets 或环境变量。"
    )

summary = portfolio_summary(st.session_state.options_holdings)
currency_symbols = {market_currency_symbol(h["market"]) for h in st.session_state.options_holdings}
summary_currency = currency_symbols.pop() if len(currency_symbols) == 1 else "$"

st.markdown(
    f"""
<div class="metric-card">
    <div class="metric-label">组合总市值</div>
    <div class="metric-value">{summary_currency}{summary['total_value']:,.2f}</div>
</div>
<div class="metric-card">
    <div class="metric-label">Short Put 接盘总市值</div>
    <div class="metric-value">{summary_currency}{summary['assignment_value']:,.2f}</div>
</div>
<div class="metric-card">
    <div class="metric-label">持仓数</div>
    <div class="metric-value">{summary['count']}</div>
</div>
""",
    unsafe_allow_html=True,
)

with st.expander("新增持仓", expanded=len(st.session_state.options_holdings) == 0):
    with st.form("add_holding_form", clear_on_submit=True):
        symbol = st.text_input("标的代码", placeholder="例如 NVDA / WDC / 00700.HK")
        position_type = st.selectbox(
            "类型",
            options=POSITION_TYPE_CHOICES,
            format_func=position_type_label,
        )
        quantity = st.number_input("数量（期权填手数，股票填股数）", min_value=1, value=1, step=1)
        strike_price = st.number_input("行权价 / 成本价", min_value=0.01, value=100.0, step=0.01)
        submitted = st.form_submit_button("添加持仓")
        if submitted:
            if not normalize_symbol(symbol):
                st.error("请输入标的代码。")
            else:
                add_holding(symbol, int(quantity), float(strike_price), position_type)
                st.success("已添加持仓。")
                st.rerun()

if st.button("更新价格", use_container_width=True):
    update_prices()
    st.rerun()

with st.expander("港元目标持仓计算器", expanded=False):
    calc_symbol = st.text_input("标的代码（用于换算）", value="NVDA")
    target_hkd = st.number_input("目标持仓（HKD）", min_value=1000.0, value=500000.0, step=1000.0)
    calc_market = infer_market(calc_symbol)
    st.caption(f"识别市场：`{calc_market.upper()}`")
    default_price = 0.0
    normalized_calc_symbol = normalize_symbol(calc_symbol)
    for h in st.session_state.options_holdings:
        if normalize_symbol(h.get("symbol", "")) == normalized_calc_symbol and isinstance(h.get("current_price"), (int, float)):
            default_price = float(h["current_price"])
            break
    price_source = st.radio(
        "价格来源",
        options=["latest", "manual"],
        format_func=lambda x: "最新价" if x == "latest" else "目标价（手动）",
        horizontal=True,
    )

    manual_price = st.number_input(
        "目标价（本币）",
        min_value=0.0,
        value=float(default_price),
        step=0.01,
        help="US 填美元价；HK 填港元价；CN 填人民币价。",
        disabled=(price_source == "latest"),
    )
    rate_col1, rate_col2 = st.columns(2)
    with rate_col1:
        usd_hkd = st.number_input("USD/HKD", min_value=1.0, value=7.80, step=0.01)
    with rate_col2:
        cny_hkd = st.number_input("CNY/HKD", min_value=0.5, value=1.08, step=0.01)

    if st.button("读取该标的最新价", use_container_width=True):
        api_key = st.session_state.get("fmp_api_key", "").strip()
        latest = fetch_single_price(calc_symbol, calc_market, api_key)
        if latest is None:
            st.warning("未获取到该标的最新价，请手动输入价格。")
        else:
            st.session_state["calc_latest_price"] = latest
            st.success(f"已获取最新价：{latest:.4f}")
            st.rerun()

    calc_price = 0.0
    if price_source == "latest":
        latest_price = st.session_state.get("calc_latest_price")
        if isinstance(latest_price, (int, float)) and latest_price > 0:
            calc_price = float(latest_price)
            st.caption(f"已使用最新价：{calc_price:.4f}")
        else:
            st.info("当前未缓存最新价，请先点击“读取该标的最新价”。")
    else:
        calc_price = float(manual_price)
        st.caption(f"已使用手动目标价：{calc_price:.4f}")

    if calc_price > 0:
        if calc_market == "us":
            hkd_per_share = calc_price * usd_hkd
        elif calc_market == "cn":
            hkd_per_share = calc_price * cny_hkd
        else:
            hkd_per_share = calc_price

        raw_shares = target_hkd / hkd_per_share if hkd_per_share > 0 else 0.0
        shares_floor = int(raw_shares)
        shares_ceil = int(raw_shares) if raw_shares.is_integer() else int(raw_shares) + 1
        short_put_lots = raw_shares / OPTION_CONTRACT_SIZE

        st.markdown(
            f"""
**换算结果**
- 每股折合：`HK${hkd_per_share:,.2f}`
- 建议持股（向下取整）：`{shares_floor:,}` 股
- 若要覆盖目标（向上取整）：`{shares_ceil:,}` 股
- Short Put 手数参考（100股/手）：`{short_put_lots:.2f}` 手（约 `~{int(round(short_put_lots))}` 手）
"""
        )
    else:
        st.info("请先输入或读取有效价格。")

st.subheader("持仓明细")
if not st.session_state.options_holdings:
    st.info("暂无持仓。先添加一条持仓。")
else:
    for item in st.session_state.options_holdings:
        symbol = item["symbol"]
        qty = float(item["quantity"])
        strike = float(item["strike_price"])
        position_type = item.get("position_type", "short_put")
        multiplier = position_multiplier(position_type)
        direction = position_direction(position_type)
        current = item.get("current_price")
        currency = market_currency_symbol(item["market"])
        distance_pct = None
        if isinstance(current, (int, float)) and strike > 0:
            distance_pct = ((float(current) - strike) / strike) * 100

        distance_text = "--" if distance_pct is None else f"{distance_pct:+.2f}%"
        assignment_value = strike * qty * multiplier * direction
        risk = position_type == "short_put" and isinstance(current, (int, float)) and current < strike
        qty_unit = "手" if multiplier == OPTION_CONTRACT_SIZE else "股"

        expander_title = (
            f"{symbol} | {position_type_label(position_type)} | "
            f"{currency}{strike:,.2f} x {int(qty)}{qty_unit}"
        )
        with st.expander(expander_title, expanded=False):
            st.write(f"市场: `{item['market'].upper()}`")
            st.write(f"最新价: `{currency}{current:,.2f}`" if isinstance(current, (int, float)) else "最新价: `--`")
            st.write(f"最新价距成本/行权价: `{distance_text}`")
            st.write(f"名义成本市值: `{currency}{assignment_value:,.2f}`")
            st.write("风险状态:")
            st.markdown(
                "<span class='pill-risk'>Short Put 已跌破行权价</span>"
                if risk
                else "<span class='pill-ok'>正常</span>",
                unsafe_allow_html=True,
            )

            with st.form(f"edit_{item['id']}"):
                new_qty = st.number_input(
                    "修改数量（手/股）",
                    min_value=1,
                    value=int(item["quantity"]),
                    step=1,
                    key=f"qty_{item['id']}",
                )
                new_strike = st.number_input(
                    "修改行权价 / 成本价",
                    min_value=0.01,
                    value=float(item["strike_price"]),
                    step=0.01,
                    key=f"strike_{item['id']}",
                )
                new_type = st.selectbox(
                    "修改类型",
                    options=POSITION_TYPE_CHOICES,
                    index=POSITION_TYPE_CHOICES.index(position_type if position_type in POSITION_TYPE_CHOICES else "short_put"),
                    format_func=position_type_label,
                    key=f"type_{item['id']}",
                )
                save_col, del_col = st.columns(2)
                with save_col:
                    save_clicked = st.form_submit_button("保存修改")
                with del_col:
                    delete_clicked = st.form_submit_button("删除持仓")

                if save_clicked:
                    item["quantity"] = int(new_qty)
                    item["strike_price"] = float(new_strike)
                    item["position_type"] = new_type
                    save_holdings_to_disk()
                    st.success("已更新。")
                    st.rerun()
                if delete_clicked:
                    remove_holding(item["id"])
                    st.warning("已删除该持仓。")
                    st.rerun()
