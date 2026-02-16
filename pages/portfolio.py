import json
import math
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
DEFAULT_PORTFOLIO_LIMIT_HKD = 30_000_000.0
DEFAULT_USD_HKD = 7.80
DEFAULT_CNY_HKD = 1.08
SUPABASE_TABLE = "portfolio_state"
SUPABASE_ROW_ID = "default"


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


def positive_int_or_default(value: object, default: int) -> int:
    try:
        parsed = int(value)
        return parsed if parsed > 0 else default
    except (TypeError, ValueError):
        return default


def item_contract_multiplier(item: Dict) -> int:
    position_type = item.get("position_type", "short_put")
    default_multiplier = position_multiplier(position_type)
    raw_multiplier = item.get("contract_multiplier")
    return positive_int_or_default(raw_multiplier, default_multiplier)


def position_direction(position_type: str) -> int:
    return -1 if position_type == "stock_sell" else 1


def market_to_hkd_rate(market: str) -> float:
    if market == "us":
        return DEFAULT_USD_HKD
    if market == "cn":
        return DEFAULT_CNY_HKD
    return 1.0


def holding_value_hkd(item: Dict, fallback_to_strike: bool = True) -> float:
    qty = float(item.get("quantity", 0) or 0)
    multiplier = item_contract_multiplier(item)
    direction = position_direction(item.get("position_type", "short_put"))
    market = item.get("market") or infer_market(item.get("symbol", ""))
    current = item.get("current_price")
    strike = float(item.get("strike_price", 0) or 0)

    price = float(current) if isinstance(current, (int, float)) else (strike if fallback_to_strike else 0.0)
    return price * qty * multiplier * direction * market_to_hkd_rate(market)


def portfolio_total_hkd(holdings: List[Dict], fallback_to_strike: bool = True) -> float:
    return sum(holding_value_hkd(item, fallback_to_strike=fallback_to_strike) for item in holdings)


def short_put_assignment_hkd(item: Dict) -> float:
    position_type = item.get("position_type", "short_put")
    if position_type != "short_put":
        return 0.0
    qty = float(item.get("quantity", 0) or 0)
    strike = float(item.get("strike_price", 0) or 0)
    market = item.get("market") or infer_market(item.get("symbol", ""))
    return strike * qty * item_contract_multiplier(item) * market_to_hkd_rate(market)


def short_put_assignment_total_hkd(holdings: List[Dict]) -> float:
    return sum(short_put_assignment_hkd(item) for item in holdings)


def option_notional_hkd(item: Dict) -> float:
    position_type = item.get("position_type", "short_put")
    if position_type not in {"short_put", "short_call"}:
        return 0.0
    qty = float(item.get("quantity", 0) or 0)
    strike = float(item.get("strike_price", 0) or 0)
    market = item.get("market") or infer_market(item.get("symbol", ""))
    return strike * qty * item_contract_multiplier(item) * market_to_hkd_rate(market)


def is_itm_short_put(item: Dict) -> bool:
    if item.get("position_type") != "short_put":
        return False
    current = item.get("current_price")
    strike = float(item.get("strike_price", 0) or 0)
    return isinstance(current, (int, float)) and float(current) <= strike


def is_itm_short_call(item: Dict) -> bool:
    if item.get("position_type") != "short_call":
        return False
    current = item.get("current_price")
    strike = float(item.get("strike_price", 0) or 0)
    return isinstance(current, (int, float)) and float(current) >= strike


def stock_value_hkd(item: Dict) -> float:
    position_type = item.get("position_type", "short_put")
    if position_type not in {"stock_long", "stock_sell"}:
        return 0.0
    market = item.get("market") or infer_market(item.get("symbol", ""))
    qty = float(item.get("quantity", 0) or 0)
    direction = position_direction(position_type)
    current = item.get("current_price")
    strike = float(item.get("strike_price", 0) or 0)
    price = float(current) if isinstance(current, (int, float)) else strike
    return price * qty * direction * market_to_hkd_rate(market)


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
    if "portfolio_limit_hkd" not in st.session_state:
        st.session_state.portfolio_limit_hkd = DEFAULT_PORTFOLIO_LIMIT_HKD
    if "include_short_side_negative" not in st.session_state:
        st.session_state.include_short_side_negative = False
    if "storage_backend" not in st.session_state:
        st.session_state.storage_backend = "local"
    if not st.session_state.holdings_loaded:
        loaded = load_holdings()
        st.session_state.options_holdings = loaded
        st.session_state.holdings_loaded = True


def supabase_credentials() -> tuple[str, str] | None:
    url = str(st.secrets.get("SUPABASE_URL", os.getenv("SUPABASE_URL", ""))).strip().rstrip("/")
    key = str(
        st.secrets.get(
            "SUPABASE_SERVICE_ROLE_KEY",
            os.getenv("SUPABASE_SERVICE_ROLE_KEY", st.secrets.get("SUPABASE_ANON_KEY", os.getenv("SUPABASE_ANON_KEY", ""))),
        )
    ).strip()
    if not url or not key:
        return None
    return url, key


def apply_loaded_payload(payload: Dict) -> List[Dict]:
    items = payload.get("holdings", [])
    last_updated = payload.get("last_updated")
    last_saved = payload.get("last_saved")
    portfolio_limit_hkd = payload.get("portfolio_limit_hkd")
    if isinstance(last_updated, str):
        st.session_state.last_updated = last_updated
    if isinstance(last_saved, str):
        st.session_state.last_saved = last_saved
    if isinstance(portfolio_limit_hkd, (int, float)):
        st.session_state.portfolio_limit_hkd = float(portfolio_limit_hkd)
    if not isinstance(items, list):
        return []

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
                "contract_multiplier": positive_int_or_default(
                    item.get("contract_multiplier"), position_multiplier(position_type)
                ),
                "current_price": item.get("current_price"),
                "created_at": item.get("created_at"),
            }
        )
    return cleaned


def load_holdings_from_supabase() -> List[Dict] | None:
    creds = supabase_credentials()
    if not creds:
        return None
    url, key = creds
    endpoint = f"{url}/rest/v1/{SUPABASE_TABLE}"
    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
    }
    params = {
        "id": f"eq.{SUPABASE_ROW_ID}",
        "select": "payload",
    }
    try:
        resp = requests.get(endpoint, headers=headers, params=params, timeout=10)
        if resp.status_code != 200:
            return None
        rows = resp.json()
        if not isinstance(rows, list) or not rows:
            return []
        payload = rows[0].get("payload")
        if not isinstance(payload, dict):
            return []
        return apply_loaded_payload(payload)
    except (requests.RequestException, ValueError):
        return None


def load_holdings_from_disk() -> List[Dict]:
    try:
        if not STORAGE_FILE.exists():
            return []
        payload = json.loads(STORAGE_FILE.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return []
        return apply_loaded_payload(payload)
    except (json.JSONDecodeError, OSError):
        return []


def load_holdings() -> List[Dict]:
    supabase_loaded = load_holdings_from_supabase()
    if supabase_loaded is not None:
        st.session_state.storage_backend = "supabase"
        return supabase_loaded
    st.session_state.storage_backend = "local"
    return load_holdings_from_disk()


def current_payload(now_str: str) -> Dict:
    return {
        "last_updated": st.session_state.last_updated,
        "last_saved": now_str,
        "portfolio_limit_hkd": float(st.session_state.get("portfolio_limit_hkd", DEFAULT_PORTFOLIO_LIMIT_HKD)),
        "holdings": st.session_state.options_holdings,
    }


def save_holdings_to_supabase(payload: Dict) -> bool:
    creds = supabase_credentials()
    if not creds:
        return False
    url, key = creds
    endpoint = f"{url}/rest/v1/{SUPABASE_TABLE}"
    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates,return=minimal",
    }
    params = {"on_conflict": "id"}
    body = [{"id": SUPABASE_ROW_ID, "payload": payload}]
    try:
        resp = requests.post(endpoint, headers=headers, params=params, json=body, timeout=10)
        return resp.status_code in {200, 201, 204}
    except requests.RequestException:
        return False


def save_holdings_to_disk() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    payload = current_payload(now_str)
    if save_holdings_to_supabase(payload):
        st.session_state.last_saved = now_str
        st.session_state.storage_backend = "supabase"
        return

    STORAGE_FILE.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    st.session_state.last_saved = now_str
    st.session_state.storage_backend = "local"


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


def portfolio_summary(holdings: List[Dict], include_short_side_negative: bool = False) -> Dict[str, float]:
    stock_long_hkd = 0.0
    stock_short_hkd = 0.0
    itm_short_put_hkd = 0.0
    itm_short_call_hkd = 0.0
    put_virtual_risk_hkd = 0.0
    call_virtual_risk_hkd = 0.0
    short_stock_virtual_hkd = 0.0
    for item in holdings:
        position_type = item.get("position_type", "short_put")
        if position_type == "stock_long":
            stock_long_hkd += abs(stock_value_hkd(item))
        elif position_type == "stock_sell":
            short_abs = abs(stock_value_hkd(item))
            stock_short_hkd += short_abs
            short_stock_virtual_hkd += short_abs
        elif position_type == "short_put":
            put_virtual_risk_hkd += option_notional_hkd(item)
            if is_itm_short_put(item):
                itm_short_put_hkd += option_notional_hkd(item)
        elif position_type == "short_call":
            call_virtual_risk_hkd += option_notional_hkd(item)
            if is_itm_short_call(item):
                itm_short_call_hkd += option_notional_hkd(item)

    if include_short_side_negative:
        total_value_hkd = stock_long_hkd + itm_short_put_hkd - itm_short_call_hkd - stock_short_hkd
        virtual_risk_hkd = put_virtual_risk_hkd - call_virtual_risk_hkd - short_stock_virtual_hkd
    else:
        total_value_hkd = stock_long_hkd + itm_short_put_hkd
        virtual_risk_hkd = put_virtual_risk_hkd

    return {
        "total_value_hkd": total_value_hkd,
        "virtual_risk_hkd": virtual_risk_hkd,
        "itm_short_put_hkd": itm_short_put_hkd,
        "itm_short_call_hkd": itm_short_call_hkd,
        "put_virtual_risk_hkd": put_virtual_risk_hkd,
        "call_virtual_risk_hkd": call_virtual_risk_hkd,
        "short_stock_virtual_hkd": short_stock_virtual_hkd,
        "stock_long_hkd": stock_long_hkd,
        "stock_short_hkd": stock_short_hkd,
        "long_bucket_hkd": stock_long_hkd + itm_short_put_hkd,
        "short_bucket_hkd": itm_short_call_hkd + stock_short_hkd,
        "count": len(holdings),
    }


def add_holding(symbol: str, quantity: int, strike_price: float, position_type: str, contract_multiplier: int) -> None:
    normalized = normalize_symbol(symbol)
    st.session_state.options_holdings.append(
        {
            "id": f"{int(datetime.now().timestamp() * 1000)}_{os.urandom(3).hex()}",
            "symbol": normalized,
            "market": infer_market(normalized),
            "quantity": int(quantity),
            "strike_price": float(strike_price),
            "position_type": position_type,
            "contract_multiplier": int(contract_multiplier),
            "current_price": None,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
    )
    save_holdings_to_disk()


def check_portfolio_limit_for_new_holding(
    symbol: str, quantity: int, strike_price: float, position_type: str, contract_multiplier: int
) -> tuple[bool, str]:
    portfolio_limit_hkd = float(st.session_state.get("portfolio_limit_hkd", DEFAULT_PORTFOLIO_LIMIT_HKD))
    current_assignment_hkd = short_put_assignment_total_hkd(st.session_state.options_holdings)
    new_item = {
        "symbol": normalize_symbol(symbol),
        "market": infer_market(symbol),
        "quantity": int(quantity),
        "strike_price": float(strike_price),
        "position_type": position_type,
        "contract_multiplier": int(contract_multiplier),
        "current_price": None,
    }
    add_assignment_hkd = short_put_assignment_hkd(new_item)
    projected_assignment_hkd = current_assignment_hkd + add_assignment_hkd

    if projected_assignment_hkd <= portfolio_limit_hkd:
        return True, ""

    if position_type != "short_put":
        return False, (
            f"当前 Short Put 接盘总市值已达 HK${current_assignment_hkd:,.0f}，超过上限 HK${portfolio_limit_hkd:,.0f}。\n"
            "可选处理：\n"
            "1) 在侧边栏提高组合上限\n"
            "2) 先降低或移除部分 Short Put 持仓"
        )

    per_qty_hkd = short_put_assignment_hkd(
        {
            "symbol": normalize_symbol(symbol),
            "market": infer_market(symbol),
            "quantity": 1,
            "strike_price": float(strike_price),
            "position_type": "short_put",
            "contract_multiplier": int(contract_multiplier),
        }
    )

    if per_qty_hkd <= 0:
        return False, (
            f"新增后预计 Short Put 接盘总市值 HK${projected_assignment_hkd:,.0f} 超过上限 HK${portfolio_limit_hkd:,.0f}。"
            "当前新增类型不会降低上限风险，请调整价格/类型。"
        )

    remaining_hkd = portfolio_limit_hkd - current_assignment_hkd
    recommended_max_qty = max(0, math.floor(remaining_hkd / per_qty_hkd))
    required_limit = projected_assignment_hkd
    return False, (
        f"新增后预计 Short Put 接盘总市值 HK${projected_assignment_hkd:,.0f}，超过上限 HK${portfolio_limit_hkd:,.0f}。\n"
        "可选处理：\n"
        f"1) 降低新增手数：把 {quantity} 调整为 {recommended_max_qty}（或更低）\n"
        f"2) 提高组合上限：至少调到 HK${required_limit:,.0f}"
    )


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

top_left, top_right = st.columns([0.7, 0.3])
with top_left:
    st.title("期权 + 正股组合")
with top_right:
    st.toggle(
        "计入空头负值",
        key="include_short_side_negative",
        help="关闭(默认)：不计入 Short Call/沽空正股；开启：作为负值计入两项市值。",
    )

last_updated = st.session_state.last_updated or "--"
st.caption(f"最后更新：{last_updated}")
st.caption(f"本地保存：{st.session_state.last_saved or '--'}")
st.caption(f"组合上限：HK${float(st.session_state.get('portfolio_limit_hkd', DEFAULT_PORTFOLIO_LIMIT_HKD)):,.0f}")

with st.sidebar:
    st.subheader("数据源设置")
    default_key = st.secrets.get("FMP_API_KEY", os.getenv("FMP_API_KEY", ""))
    st.session_state.fmp_api_key = st.text_input(
        "FMP API Key", value=default_key, type="password", help="优先读取 secrets 或环境变量。"
    )
    limit_val = st.number_input(
        "组合上限 (HKD)",
        min_value=100_000.0,
        value=float(st.session_state.get("portfolio_limit_hkd", DEFAULT_PORTFOLIO_LIMIT_HKD)),
        step=100_000.0,
    )
    if float(limit_val) != float(st.session_state.get("portfolio_limit_hkd", DEFAULT_PORTFOLIO_LIMIT_HKD)):
        st.session_state.portfolio_limit_hkd = float(limit_val)
        save_holdings_to_disk()
    backend = st.session_state.get("storage_backend", "local")
    backend_text = "Supabase 持久化" if backend == "supabase" else "本地文件（Reboot 会丢失）"
    st.caption(f"存储方式：{backend_text}")

summary = portfolio_summary(
    st.session_state.options_holdings,
    include_short_side_negative=bool(st.session_state.get("include_short_side_negative", False)),
)

st.markdown(
    f"""
<div class="metric-card">
    <div class="metric-label">组合总市值 (风险调整, HKD)</div>
    <div class="metric-value">HK${summary['total_value_hkd']:,.2f}</div>
</div>
<div class="metric-card">
    <div class="metric-label">虚拟风险市值 (HKD)</div>
    <div class="metric-value">HK${summary['virtual_risk_hkd']:,.2f}</div>
</div>
<div class="metric-card">
    <div class="metric-label">持仓数</div>
    <div class="metric-value">{summary['count']}</div>
</div>
""",
    unsafe_allow_html=True,
)

with st.expander("市值拆分明细 (HKD)", expanded=False):
    mode_text = "已计入 Short Call + 沽空正股负值" if st.session_state.get("include_short_side_negative") else "未计入 Short Call + 沽空正股负值（默认）"
    st.markdown(
        f"""
**当前口径**
- `{mode_text}`

**组合总市值拆分**
- `Short Put + 正股(买入)`：`HK${summary['long_bucket_hkd']:,.2f}`
  - 其中 正股(买入)：`HK${summary['stock_long_hkd']:,.2f}`
  - 其中 入价 Short Put：`HK${summary['itm_short_put_hkd']:,.2f}`
- `Short Call + 沽空正股`：`HK${summary['short_bucket_hkd']:,.2f}`
  - 其中 入价 Short Call：`HK${summary['itm_short_call_hkd']:,.2f}`
  - 其中 沽空正股：`HK${summary['stock_short_hkd']:,.2f}`

**虚拟风险市值拆分**
- `Short Put 名义风险`：`HK${summary['put_virtual_risk_hkd']:,.2f}`
- `Short Call 名义风险`：`HK${summary['call_virtual_risk_hkd']:,.2f}`
- `沽空正股虚拟值`：`HK${summary['short_stock_virtual_hkd']:,.2f}`
- `最终虚拟风险市值`：`HK${summary['virtual_risk_hkd']:,.2f}`
"""
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
        contract_multiplier = st.number_input(
            "期权乘数（默认100，可手动修改）",
            min_value=1,
            value=OPTION_CONTRACT_SIZE,
            step=1,
            help="仅对 Short Put / Short Call 生效；正股类型固定按 1 计算。",
            disabled=(position_type in {"stock_long", "stock_sell"}),
        )
        strike_price = st.number_input("行权价 / 成本价", min_value=0.01, value=100.0, step=0.01)
        submitted = st.form_submit_button("添加持仓")
        if submitted:
            if not normalize_symbol(symbol):
                st.error("请输入标的代码。")
            else:
                effective_multiplier = int(contract_multiplier) if position_type in {"short_put", "short_call"} else 1
                allowed, warning_msg = check_portfolio_limit_for_new_holding(
                    symbol, int(quantity), float(strike_price), position_type, effective_multiplier
                )
                if not allowed:
                    st.warning(warning_msg)
                else:
                    add_holding(symbol, int(quantity), float(strike_price), position_type, effective_multiplier)
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
        multiplier = item_contract_multiplier(item)
        direction = position_direction(position_type)
        current = item.get("current_price")
        currency = market_currency_symbol(item["market"])
        distance_pct = None
        if isinstance(current, (int, float)) and strike > 0:
            distance_pct = ((float(current) - strike) / strike) * 100

        distance_text = "--" if distance_pct is None else f"{distance_pct:+.2f}%"
        assignment_value = strike * qty * multiplier * direction
        virtual_risk = option_notional_hkd(item)
        risk_put = is_itm_short_put(item)
        risk_call = is_itm_short_call(item)
        qty_unit = "手" if position_type in {"short_put", "short_call"} else "股"

        expander_title = (
            f"{symbol} | {position_type_label(position_type)} | "
            f"{currency}{strike:,.2f} x {int(qty)}{qty_unit}"
        )
        with st.expander(expander_title, expanded=False):
            st.write(f"市场: `{item['market'].upper()}`")
            st.write(f"最新价: `{currency}{current:,.2f}`" if isinstance(current, (int, float)) else "最新价: `--`")
            st.write(f"最新价距成本/行权价: `{distance_text}`")
            st.write(f"名义成本市值: `{currency}{assignment_value:,.2f}`")
            st.write(f"合约乘数: `{multiplier}`")
            st.write(f"虚拟风险市值(HKD): `HK${virtual_risk:,.2f}`")
            st.write("风险状态:")
            st.markdown(
                "<span class='pill-risk'>Short Put 已跌破行权价</span>"
                if risk_put
                else ("<span class='pill-risk'>Short Call 已升破行权价</span>" if risk_call else "<span class='pill-ok'>正常</span>"),
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
                new_multiplier = st.number_input(
                    "修改期权乘数",
                    min_value=1,
                    value=int(item_contract_multiplier(item)),
                    step=1,
                    key=f"multiplier_{item['id']}",
                    help="仅对 Short Put / Short Call 生效；正股类型固定按 1 计算。",
                    disabled=(new_type in {"stock_long", "stock_sell"}),
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
                    item["contract_multiplier"] = int(new_multiplier) if new_type in {"short_put", "short_call"} else 1
                    save_holdings_to_disk()
                    st.success("已更新。")
                    st.rerun()
                if delete_clicked:
                    remove_holding(item["id"])
                    st.warning("已删除该持仓。")
                    st.rerun()
