#!/usr/bin/env python3
"""
Signal Monitor — Zero-AI autonomous signal checker + Discord alerter + dashboard
=================================================================================
systemd timer every 30 min. Zero AI tokens.
"""

import json, sys, hashlib, requests
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from momentum_signals import check_all_signals
from daily_signals import check_daily_signals
from signal_tracker import update_signals
from signal_charts import generate_all_signal_charts

DATA_DIR = SCRIPT_DIR / "data" / "signals"
STATE_FILE = DATA_DIR / "state.json"
HISTORY_FILE = DATA_DIR / "history.jsonl"
DASHBOARD_FILE = DATA_DIR / "dashboard.html"
OPENCLAW_CONFIG = Path.home() / ".openclaw" / "openclaw.json"
DISCORD_SIGNALS_CHANNEL = "1469863344910631025"
DISCORD_API = "https://discord.com/api/v10"


def get_discord_token():
    try:
        return json.loads(OPENCLAW_CONFIG.read_text()).get("channels", {}).get("discord", {}).get("token")
    except Exception:
        return None


def discord_post(channel_id, content, token):
    try:
        r = requests.post(f"{DISCORD_API}/channels/{channel_id}/messages",
                          headers={"Authorization": f"Bot {token}", "Content-Type": "application/json"},
                          json={"content": content}, timeout=10)
        return r.status_code in (200, 201)
    except Exception as e:
        print(f"Discord: {e}"); return False


def check_all():
    now = datetime.now(timezone.utc)
    try: history, stats = update_signals()
    except: history, stats = [], {}
    try: momentum = check_all_signals()
    except: momentum = {"signals": [], "no_signal": []}
    try: daily = check_daily_signals()
    except: daily = {"signals": [], "no_signal": []}

    sigs = (momentum.get("signals") or []) + (daily.get("signals") or [])
    nosig = (momentum.get("no_signal") or []) + (daily.get("no_signal") or [])
    return {
        "timestamp": now.isoformat(), "timestamp_unix": int(now.timestamp()),
        "status": "signals" if sigs else "ok", "signals": sigs, "no_signal": nosig,
        "tracker_stats": stats, "tracker_history": history, "signal_count": len(sigs),
    }


def signal_fingerprint(signals):
    parts = sorted(f"{s.get('asset','')}-{s.get('direction','')}-{s.get('signal_type','')}" for s in signals)
    return hashlib.md5("|".join(parts).encode()).hexdigest()


def detect_changes(prev, curr):
    if prev is None:
        return [f"🆕 Initial: {', '.join(s['asset']+' '+s['direction'] for s in curr['signals'])}"] if curr["signals"] else []
    if signal_fingerprint(prev.get("signals", [])) == signal_fingerprint(curr.get("signals", [])):
        return []
    changes = []
    pm = {s["asset"]: s for s in prev.get("signals", [])}
    cm = {s["asset"]: s for s in curr.get("signals", [])}
    for a, s in cm.items():
        if a not in pm: changes.append(f"🆕 NEW: {s['direction']} {a}")
        elif pm[a]["direction"] != s["direction"]: changes.append(f"🔄 FLIP: {a} {pm[a]['direction']}→{s['direction']}")
    for a in pm:
        if a not in cm: changes.append(f"❌ GONE: {a}")
    return changes


def format_discord_alert(state, changes):
    now = datetime.fromisoformat(state["timestamp"])
    lines = [f"📡 **Signal Update — {now.strftime('%b %d, %H:%M UTC')}**", ""]
    for c in changes: lines.append(f"  {c}")
    if changes: lines.append("")
    for sig in state.get("signals", []):
        d = "🟢" if sig["direction"] == "LONG" else "🔴"
        t = {"jump_fade":"Jump Fade","jump_trend":"Jump Trend","hurst_trend":"Hurst Trend","hurst_fade":"Hurst Fade"}.get(sig.get("signal_type",""), sig.get("signal_type",""))
        e = sig.get("entry", 0)
        f = ",.2f" if sig.get("symbol") == "GC=F" else (".4f" if e > 1 else ".6f")
        lines.append(f"{d} **{sig['direction']} {sig['asset']}** — {t} | Entry ${e:{f}} | SL ${sig.get('stop_loss',0):{f}} | TP ${sig.get('take_profit',0):{f}}")
        if sig.get("details"): lines.append(f"  └ {sig['details']}")
    return "\n".join(lines)


def load_check_history(n=500):
    if not HISTORY_FILE.exists(): return []
    entries = []
    for line in HISTORY_FILE.read_text().splitlines():
        if line.strip():
            try: entries.append(json.loads(line))
            except: pass
    return entries[-n:]


def generate_dashboard(state, chart_files):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    check_hist = load_check_history(200)
    tracker_hist = state.get("tracker_history", [])
    stats = state.get("tracker_stats", {})

    # Equity curve
    equity = []
    cum = 0
    for th in sorted(tracker_hist, key=lambda x: x.get("timestamp", "")):
        if th.get("outcome") in ("WIN", "LOSS"):
            cum += th.get("actual_pnl_usd", 0)
            equity.append({"time": th.get("exit_time", th.get("timestamp", "")), "pnl": round(cum, 2),
                           "outcome": th.get("outcome"), "asset": th.get("asset", "")})

    data = json.dumps({
        "generated": state.get("timestamp", ""),
        "signals": state.get("signals", []),
        "stats": stats,
        "chart_files": chart_files,
        "equity_curve": equity,
        "check_history": check_hist,
        "tracker_history": [
            {k: v for k, v in th.items() if k in (
                "asset","symbol","direction","entry","stop_loss","take_profit",
                "signal_type","outcome","actual_pnl_usd","r_multiple",
                "time_to_resolution_hours","timestamp","exit_time")}
            for th in tracker_hist],
    }, default=str)

    DASHBOARD_FILE.write_text(TEMPLATE.replace("__DATA__", data))


TEMPLATE = r"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Curupira Signals</title>
<style>
:root{--bg:#0a0a0f;--bg2:#0f0f1a;--bg3:#0a0a14;--border:#1a1a2e;--text:#c8c8d0;--dim:#666;--green:#00ff88;--red:#ff4444;--blue:#4488ff;--yellow:#ffaa00;--font:'JetBrains Mono','Fira Code','Consolas',monospace}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:var(--font);background:var(--bg);color:var(--text);padding:16px}
.hdr{display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid var(--border);padding-bottom:12px;margin-bottom:16px}
.hdr h1{color:var(--green);font-size:1.3em}
.hdr .meta{color:var(--dim);font-size:.8em;text-align:right}
.badge{display:inline-block;padding:3px 10px;border-radius:3px;font-size:.8em;font-weight:600}
.badge-on{background:#1a3a1a;color:var(--green);border:1px solid #00ff8844}
.badge-off{background:#1a1a2e;color:var(--dim);border:1px solid #333}
.stats{display:flex;gap:10px;margin-bottom:16px;flex-wrap:wrap}
.sb{background:var(--bg2);border:1px solid var(--border);border-radius:6px;padding:10px 14px;text-align:center;flex:1;min-width:90px}
.sb .v{font-size:1.4em;font-weight:700}
.sb .l{font-size:.65em;color:var(--dim);text-transform:uppercase;letter-spacing:1px;margin-top:2px}
.vg{color:var(--green)}.vr{color:var(--red)}.vb{color:var(--blue)}.vy{color:var(--yellow)}
.charts{margin-bottom:16px}
.charts img{width:100%;max-width:100%;border-radius:6px;border:1px solid var(--border);margin-bottom:12px}
.card{background:var(--bg2);border:1px solid var(--border);border-radius:6px;margin-bottom:16px;overflow:hidden}
.card .t{padding:8px 12px;font-size:.85em;color:var(--dim);border-bottom:1px solid var(--border)}
table{width:100%;border-collapse:collapse;font-size:.8em}
th{text-align:left;color:var(--dim);font-weight:500;padding:6px 10px;border-bottom:1px solid var(--border)}
td{padding:6px 10px;border-bottom:1px solid #0a0a14}
tr:hover td{background:var(--bg3)}
.tl{color:var(--green)}.ts{color:var(--red)}.tw{color:var(--green);font-weight:600}.tlo{color:var(--red);font-weight:600}.to{color:var(--yellow)}
.empty{color:#333;text-align:center;padding:30px;font-style:italic}
.eq-empty{background:var(--bg2);border:1px solid var(--border);border-radius:6px;padding:40px;text-align:center;color:#333;margin-bottom:16px}
.footer{text-align:center;color:#222;margin-top:30px;font-size:.7em}
</style></head><body>
<script>const D=__DATA__;
const s=D.stats||{};const pnl=s.total_pnl_usd||0;const wr=s.win_rate||0;
const pf=s.profit_factor;const pfS=pf===Infinity?'∞':(pf||'-');
document.write(`
<div class="hdr"><h1>🌿 Curupira Signals</h1><div class="meta">
${new Date(D.generated).toLocaleString()}<br>
<span class="badge ${D.signals.length?'badge-on':'badge-off'}">${D.signals.length?D.signals.length+' ACTIVE':'NO SIGNALS'}</span>
</div></div>
<div class="stats">
<div class="sb"><div class="v vb">${s.total_signals||0}</div><div class="l">Total</div></div>
<div class="sb"><div class="v vg">${s.wins||0}</div><div class="l">Wins</div></div>
<div class="sb"><div class="v vr">${s.losses||0}</div><div class="l">Losses</div></div>
<div class="sb"><div class="v ${wr>=50?'vg':wr>0?'vy':'vb'}">${wr}%</div><div class="l">Win Rate</div></div>
<div class="sb"><div class="v ${pf>=1.5?'vg':pf>=1?'vy':'vr'}">${pfS}</div><div class="l">PF</div></div>
<div class="sb"><div class="v ${pnl>=0?'vg':'vr'}">$${pnl.toLocaleString('en',{minimumFractionDigits:2})}</div><div class="l">P&L</div></div>
<div class="sb"><div class="v vb">${s.avg_r||'-'}</div><div class="l">Avg R</div></div>
<div class="sb"><div class="v ${s.open?'vy':'vb'}">${s.open||0}</div><div class="l">Open</div></div>
</div>`);

// Charts (JPEGs)
const cf=D.chart_files||{};const chartKeys=Object.keys(cf);
if(chartKeys.length){
document.write('<div class="charts">');
chartKeys.forEach(k=>{document.write(`<img src="${cf[k]}" alt="${k} signal chart" loading="lazy">`)});
document.write('</div>');
}else if(!D.signals.length){
document.write('<div class="eq-empty">No active signals — charts appear when signals fire</div>');
}

// Equity curve placeholder
if(D.equity_curve&&D.equity_curve.length){
document.write('<div class="card"><div class="t">📈 Equity Curve</div><div style="padding:12px">');
D.equity_curve.forEach(e=>{
const cls=e.outcome==='WIN'?'tw':'tlo';
document.write(`<span class="${cls}" style="margin-right:12px">${e.asset} $${e.pnl.toFixed(2)}</span>`);
});
document.write('</div></div>');
}

// Trade history
const th=D.tracker_history||[];
if(th.length){
document.write('<div class="card"><div class="t">📋 Trade History</div><table><thead><tr><th>Time</th><th>Asset</th><th>Dir</th><th>Type</th><th>Entry</th><th>Result</th><th>P&L</th><th>R</th></tr></thead><tbody>');
th.slice().reverse().forEach(t=>{
const dc=t.direction==='LONG'?'tl':'ts';
const oc=t.outcome==='WIN'?'tw':(t.outcome==='LOSS'?'tlo':'to');
const pnl=t.actual_pnl_usd!=null?'$'+t.actual_pnl_usd.toFixed(2):'-';
document.write(`<tr><td>${(t.timestamp||'').slice(0,16)}</td><td>${t.asset||''}</td><td class="${dc}">${t.direction||''}</td><td>${t.signal_type||''}</td><td>$${(t.entry||0).toFixed(4)}</td><td class="${oc}">${t.outcome||'OPEN'}</td><td class="${oc}">${pnl}</td><td>${t.r_multiple||'-'}</td></tr>`);
});
document.write('</tbody></table></div>');
}

// Check history (48h)
const cutoff=Date.now()/1000-48*3600;
const recent=(D.check_history||[]).filter(h=>(h.timestamp_unix||0)>cutoff).reverse().slice(0,30);
if(recent.length){
document.write('<div class="card"><div class="t">🕐 Signal Checks (48h)</div><table><thead><tr><th>Time</th><th>Dir</th><th>Asset</th><th>Type</th><th>Entry</th></tr></thead><tbody>');
recent.forEach(h=>{
const t=new Date(h.timestamp).toLocaleString([],{month:'short',day:'numeric',hour:'2-digit',minute:'2-digit'});
const sigs=h.signals||[];
if(sigs.length){sigs.forEach(s=>{
document.write(`<tr><td>${t}</td><td class="${s.direction==='LONG'?'tl':'ts'}">${s.direction}</td><td>${s.asset}</td><td>${s.signal_type||''}</td><td>$${(s.entry||0).toFixed(4)}</td></tr>`);
})}else{document.write(`<tr><td>${t}</td><td colspan="4" style="color:#333">—</td></tr>`)}
});
document.write('</tbody></table></div>');
}

document.write('<div class="footer">signal_monitor.py — zero AI tokens 🔥</div>');
</script></body></html>"""


def main():
    print(f"[{datetime.now().isoformat()}] Signal monitor starting...")
    state = check_all()
    print(f"  Status: {state['status']} | Signals: {state['signal_count']}")

    prev = load_previous_state()
    changes = detect_changes(prev, state)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    # Save state (slim)
    slim = {k: v for k, v in state.items() if k != "tracker_history"}
    STATE_FILE.write_text(json.dumps(slim, indent=2, default=str))
    with HISTORY_FILE.open("a") as f:
        f.write(json.dumps(slim, default=str) + "\n")

    if changes:
        print(f"  Changes: {', '.join(changes)}")
        token = get_discord_token()
        if token:
            ok = discord_post(DISCORD_SIGNALS_CHANNEL, format_discord_alert(state, changes), token)
            print(f"  Discord: {'ok' if ok else 'FAIL'}")
    else:
        print("  No changes")

    print("  Generating charts...")
    chart_files = generate_all_signal_charts(state.get("signals", []), DATA_DIR) if state["signals"] else {}
    print(f"  Charts: {len(chart_files)}")

    generate_dashboard(state, chart_files)
    print(f"  Dashboard: {DASHBOARD_FILE}")
    print(f"[{datetime.now().isoformat()}] Done")


def load_previous_state():
    try: return json.loads(STATE_FILE.read_text())
    except: return None


if __name__ == "__main__":
    main()
