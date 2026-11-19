#!/usr/bin/env python3
"""CHoCH gate study. Pre-registered spec: see REPORT.md header.
Reuses div12m event detection / exits / cost / split verbatim (same code paths).
Variants: A = entry at divergence confirmation close (div12m baseline);
B = entry at CHoCH confirmation bar close; C = entry at first pullback touch close after CHoCH.
CHoCH (long): most recent CONFIRMED price swing high (pivot +-3, pivot idx p with p+3 <= conf idx)
before the divergence confirmation bar; a later closed bar closing strictly above it within
20 bars after conf = CHoCH. Short mirrored. No lookahead anywhere.
"""
import csv, json, os
from datetime import datetime, timezone
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
DIV_EVENTS = os.path.expanduser("~/workspace/gate-exec/div12m_20261119/events.csv")
END_TS = 1795046400000          # 2026-11-19 00:00 UTC
CAL_END = 1787097600000         # 2026-08-19 00:00 UTC
HOLD_START = 1787184000000      # 2026-08-20 00:00 UTC
MAXHOLD = 200
COST_RT = 0.002
CHOCH_WINDOW = 20
PULLBACK_WINDOW = 10
ZONE_ATR = 0.25

GROUPS = [("BTC", "30m", "btc_30m.csv"), ("BTC", "15m", "btc_15m.csv"), ("BTC", "1H", "btc_1H.csv"),
          ("ETH", "30m", "eth_30m.csv"), ("SOL", "30m", "sol_30m.csv"), ("DOGE", "30m", "doge_30m.csv")]

# ---------- div12m primitives (verbatim) ----------
def ema(x, n):
    k = 2.0 / (n + 1.0)
    out = np.empty_like(x)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = out[i - 1] + k * (x[i] - out[i - 1])
    return out

def atr14(h, l, c):
    n = len(c)
    tr = np.empty(n)
    tr[0] = h[0] - l[0]
    for i in range(1, n):
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
    out = np.empty(n)
    out[:14] = np.cumsum(tr[:14]) / np.arange(1, 15)
    a = tr[:14].mean()
    out[13] = a
    for i in range(14, n):
        a = a + (tr[i] - a) / 14.0
        out[i] = a
    return out

def pivots(arr, w=3):
    idx = []
    n = len(arr)
    for i in range(w, n - w):
        seg = arr[i - w:i + w + 1]
        if arr[i] == seg.max() and (seg[:w] < arr[i]).all() and (seg[w + 1:] <= arr[i]).all():
            idx.append(i)
    return idx

def pivots_min(arr, w=3):
    idx = []
    n = len(arr)
    for i in range(w, n - w):
        seg = arr[i - w:i + w + 1]
        if arr[i] == seg.min() and (seg[:w] > arr[i]).all() and (seg[w + 1:] >= arr[i]).all():
            idx.append(i)
    return idx

def exit_fixed(h, l, c, i0, entry, sl, tp, direction, maxhold=MAXHOLD):
    end = min(len(c), i0 + maxhold)
    for j in range(i0, end):
        if direction == 1:
            if l[j] <= sl: return sl
            if h[j] >= tp: return tp
        else:
            if h[j] >= sl: return sl
            if l[j] <= tp: return tp
    return c[end - 1]

def exit_trail(h, l, c, i0, entry, sl, R, direction, trail_dist, maxhold=MAXHOLD):
    end = min(len(c), i0 + maxhold)
    peak = entry
    stop = sl
    for j in range(i0, end):
        if direction == 1:
            if l[j] <= stop: return stop
            if h[j] > peak:
                peak = h[j]
                if (peak - entry) / R >= 1.0:
                    stop = max(stop, peak - trail_dist * R)
        else:
            if h[j] >= stop: return stop
            if l[j] < peak:
                peak = l[j]
                if (entry - peak) / R >= 1.0:
                    stop = min(stop, peak + trail_dist * R)
    return c[end - 1]

def exit_timestop(h, l, c, i0, entry, sl, tp, direction, bars):
    end = min(len(c), i0 + bars)
    for j in range(i0, end):
        if direction == 1:
            if l[j] <= sl: return sl
            if h[j] >= tp: return tp
        else:
            if h[j] >= sl: return sl
            if l[j] <= tp: return tp
    return c[end - 1]

def metrics(rs, ws):
    rs = np.asarray(rs, float); ws = np.asarray(ws, float)
    if len(rs) == 0:
        return dict(n=0, win=None, avg=None, pf=None, wavg=None, wpf=None)
    win = float((rs > 0).mean())
    gp = rs[rs > 0].sum(); gl = -rs[rs < 0].sum()
    pf = float(gp / gl) if gl > 0 else (float("inf") if gp > 0 else None)
    wgp = (ws[rs > 0] * rs[rs > 0]).sum(); wgl = -(ws[rs < 0] * rs[rs < 0]).sum()
    wpf = float(wgp / wgl) if wgl > 0 else (float("inf") if wgp > 0 else None)
    return dict(n=int(len(rs)), win=win, avg=float(rs.mean()), pf=pf,
                wavg=float((ws * rs).sum() / ws.sum()), wpf=wpf)

# ---------- CHoCH machinery ----------
def last_confirmed_pivot(piv_list, conf):
    """Largest pivot index p with p+3 <= conf (pivot known by confirmation bar)."""
    best = None
    for p in piv_list:
        if p + 3 <= conf:
            best = p
        else:
            break
    return best

def eval_variant(h, l, c, eidx, entry, sl, direction):
    """Return dict of net/gross R for exit rules, exits scanned from eidx+1 (div12m convention)."""
    R = abs(entry - sl)
    if R <= 0 or eidx + 1 >= len(c):
        return None
    cost_r = COST_RT * entry / R
    i0 = eidx + 1
    out = {}
    tp = entry + direction * 2.0 * R
    xp = exit_fixed(h, l, c, i0, entry, sl, tp, direction)
    out["tp2"] = direction * (xp - entry) / R - cost_r
    out["tp2_gross"] = direction * (xp - entry) / R
    xp = exit_trail(h, l, c, i0, entry, sl, R, direction, 0.3)
    out["trail03"] = direction * (xp - entry) / R - cost_r
    for nb, key in ((48, "ts48"), (96, "ts96")):
        tp3 = entry + direction * 3.0 * R
        xp = exit_timestop(h, l, c, i0, entry, sl, tp3, direction, nb)
        out[key] = direction * (xp - entry) / R - cost_r
    out["R"] = float(R); out["cost_r"] = float(cost_r)
    return out

def run_group(sym, bar, fname):
    path = os.path.join(DATA, fname)
    ts, o, h, l, c = [], [], [], [], []
    with open(path) as f:
        rd = csv.reader(f); next(rd)
        for row in rd:
            ts.append(int(row[0])); o.append(float(row[1])); h.append(float(row[2]))
            l.append(float(row[3])); c.append(float(row[4]))
    ts = np.array(ts); o = np.array(o); h = np.array(h); l = np.array(l); c = np.array(c)
    n = len(c)
    macd = ema(c, 12) - ema(c, 26)
    hist = macd - ema(macd, 9)
    atr = atr14(h, l, c)
    low_piv = pivots_min(l)
    high_piv = pivots(h)
    rows = []   # one dict per (event, variant)
    events_meta = []

    def process(piv_list, is_low):
        for a, b in zip(piv_list, piv_list[1:]):
            if is_low:
                if not (l[b] < l[a] and hist[b] > hist[a]): continue
                direction = 1
            else:
                if not (h[b] > h[a] and hist[b] < hist[a]): continue
                direction = -1
            conf = b + 3
            if conf >= n: continue
            a14 = atr[b]
            if a14 <= 0: continue
            strength = (hist[b] - hist[a]) / a14 if direction == 1 else (hist[a] - hist[b]) / a14
            ext = l[b] if is_low else h[b]
            sl = ext - 0.1 * a14 if direction == 1 else ext + 0.1 * a14
            w = 0.5 ** (((END_TS - ts[conf]) / 86400000.0) / 90.0)
            split = ("cal" if ts[conf] < CAL_END else ("hold" if ts[conf] >= HOLD_START else "gap"))
            ev = dict(sym=sym, tf=bar, conf_ts=int(ts[conf]), direction=direction,
                      strength=float(strength), sl=float(sl), w=float(w), split=split)
            # --- CHoCH structure level: most recent confirmed opposite pivot before conf ---
            opp = high_piv if direction == 1 else low_piv
            p = last_confirmed_pivot(opp, conf)
            level = None
            if p is not None:
                level = float(h[p]) if direction == 1 else float(l[p])
            ev["level"] = level
            # --- CHoCH scan: bars conf+1 .. conf+20, close strictly breaks level ---
            choch_idx = None
            if level is not None:
                for j in range(conf + 1, min(n, conf + 1 + CHOCH_WINDOW)):
                    if (direction == 1 and c[j] > level) or (direction == -1 and c[j] < level):
                        choch_idx = j
                        break
            ev["choch"] = choch_idx is not None
            ev["choch_delay"] = (choch_idx - conf) if choch_idx is not None else None
            ev["pre_sl_touch"] = None
            if choch_idx is not None:
                seg = range(conf + 1, choch_idx + 1)
                if direction == 1:
                    ev["pre_sl_touch"] = bool(any(l[j] <= sl for j in seg))
                else:
                    ev["pre_sl_touch"] = bool(any(h[j] >= sl for j in seg))
            events_meta.append(ev)
            base = dict(sym=sym, tf=bar, conf_ts=int(ts[conf]),
                        direction=("long" if direction == 1 else "short"),
                        strength=float(strength), split=split, w=float(w),
                        level=level, choch=ev["choch"], choch_delay=ev["choch_delay"],
                        pre_sl_touch=ev["pre_sl_touch"])
            # --- Variant A: baseline entry at conf close ---
            entryA = float(c[conf])
            rA = eval_variant(h, l, c, conf, entryA, sl, direction)
            rowA = dict(base, variant="A", status="entered" if rA else "invalid",
                        entry_ts=int(ts[conf]), entry=entryA, sl=float(sl))
            if rA: rowA.update(rA)
            rows.append(rowA)
            # --- Variant B: entry at CHoCH bar close ---
            if choch_idx is None:
                rows.append(dict(base, variant="B", status="no_choch", entry_ts=None, entry=None, sl=float(sl)))
            else:
                entryB = float(c[choch_idx])
                rB = eval_variant(h, l, c, choch_idx, entryB, sl, direction)
                rowB = dict(base, variant="B", status="entered" if rB else "invalid",
                            entry_ts=int(ts[choch_idx]), entry=entryB, sl=float(sl))
                if rB: rowB.update(rB)
                rows.append(rowB)
                # --- Variant C: pullback entry within 10 bars after CHoCH ---
                zone = ZONE_ATR * atr[choch_idx]
                statusC, entry_idx = "no_pullback", None
                for j in range(choch_idx + 1, min(n, choch_idx + 1 + PULLBACK_WINDOW)):
                    if direction == 1:
                        if c[j] < sl:
                            statusC = "invalidated"; break
                        if l[j] <= level + zone:
                            entry_idx = j; break
                    else:
                        if c[j] > sl:
                            statusC = "invalidated"; break
                        if h[j] >= level - zone:
                            entry_idx = j; break
                if entry_idx is not None:
                    entryC = float(c[entry_idx])
                    rC = eval_variant(h, l, c, entry_idx, entryC, sl, direction)
                    rowC = dict(base, variant="C", status="entered" if rC else "invalid",
                                entry_ts=int(ts[entry_idx]), entry=entryC, sl=float(sl))
                    if rC: rowC.update(rC)
                    rows.append(rowC)
                else:
                    rows.append(dict(base, variant="C", status=statusC, entry_ts=None, entry=None, sl=float(sl)))
            if choch_idx is None:
                rows.append(dict(base, variant="C", status="no_choch", entry_ts=None, entry=None, sl=float(sl)))
    process(low_piv, True)
    process(high_piv, False)
    return rows, events_meta

def agg(rows, variant, split, key="tp2"):
    sub = [r for r in rows if r["variant"] == variant and r["split"] == split and r["status"] == "entered"]
    m = metrics([r[key] for r in sub], [r["w"] for r in sub])
    g = metrics([r["tp2_gross"] for r in sub], [r["w"] for r in sub]) if key == "tp2" else None
    return m, g

def pass_check(cal_m, hold_m):
    return bool(hold_m["n"] >= 20 and hold_m["pf"] is not None and hold_m["pf"] >= 1.2
                and hold_m["avg"] is not None and hold_m["avg"] >= 0.15
                and cal_m["wpf"] is not None and cal_m["wpf"] >= 1.2)

def main():
    all_rows, all_meta = [], []
    group_rows = {}
    for sym, bar, fn in GROUPS:
        rows, meta = run_group(sym, bar, fn)
        group_rows[(sym, bar)] = rows
        all_rows.extend(rows); all_meta.extend(meta)
        print(f"{sym} {bar}: {len(meta)} events", flush=True)
    # quintile assignment per (sym, tf) from CAL strengths (div12m method)
    for (sym, bar), rows in list(group_rows.items()):
        cal_s = [r["strength"] for r in rows if r["variant"] == "A" and r["split"] == "cal"]
        qs = np.quantile(cal_s, [0.2, 0.4, 0.6, 0.8]) if len(cal_s) >= 25 else np.zeros(4)
        for r in rows:
            if r["sym"] == sym and r["tf"] == bar:
                r["quintile"] = int(np.searchsorted(qs, r["strength"], side="right")) + 1
        group_rows[(sym, bar) + ("qs",)] = qs.tolist()
    pooled30 = [r for k, rows in group_rows.items() if len(k) == 2 and k[1] == "30m" for r in rows]
    # pooled quintiles from pooled CAL strengths
    cal_s = [r["strength"] for r in pooled30 if r["variant"] == "A" and r["split"] == "cal"]
    pqs = np.quantile(cal_s, [0.2, 0.4, 0.6, 0.8])
    for r in pooled30:
        r["quintile_pooled"] = int(np.searchsorted(pqs, r["strength"], side="right")) + 1

    results = {}
    def group_block(name, rows):
        blk = {}
        for v in "ABC":
            blk[v] = {}
            for sp in ("cal", "hold"):
                m, g = agg(rows, v, sp)
                blk[v][sp] = dict(net=m, gross=(g if sp else None))
            _, gh = agg(rows, v, "hold")
            blk[v]["hold_gross"] = gh
            for key in ("trail03", "ts48", "ts96"):
                mh, _ = agg(rows, v, "hold", key)
                blk[v][f"hold_{key}"] = mh
        blk["pass"] = {v: pass_check(blk[v]["cal"]["net"], blk[v]["hold"]["net"]) for v in "BC"}
        results[name] = blk
        return blk
    for k, rows in group_rows.items():
        if len(k) == 2:
            group_block(f"{k[0]}_{k[1]}", rows)
    group_block("POOLED_30m", pooled30)

    # CHoCH stats
    def choch_stats(meta):
        tot = len(meta); conf = [m for m in meta if m["choch"]]
        delays = [m["choch_delay"] for m in conf]
        return dict(n_events=tot, n_choch=len(conf),
                    rate=(len(conf) / tot if tot else None),
                    delay_median=(float(np.median(delays)) if delays else None),
                    delay_mean=(float(np.mean(delays)) if delays else None),
                    pre_sl_touch_rate=(float(np.mean([m["pre_sl_touch"] for m in conf])) if conf else None))
    results["choch_stats"] = {
        "BTC_30m": choch_stats([m for m in all_meta if m["sym"] == "BTC" and m["tf"] == "30m"]),
        "POOLED_30m": choch_stats([m for m in all_meta if m["tf"] == "30m"]),
        "BTC_30m_cal": choch_stats([m for m in all_meta if m["sym"] == "BTC" and m["tf"] == "30m" and m["split"] == "cal"]),
        "BTC_30m_hold": choch_stats([m for m in all_meta if m["sym"] == "BTC" and m["tf"] == "30m" and m["split"] == "hold"]),
    }
    # Same-subset comparison (events WITH choch): A vs B vs C per-trade quality, BTC30 + pooled
    def subset_block(rows):
        keys = set((r["sym"], r["tf"], r["conf_ts"], r["direction"]) for r in rows if r["choch"])
        out = {}
        for v in "ABC":
            out[v] = {}
            for sp in ("cal", "hold"):
                sub = [r for r in rows if r["variant"] == v and r["split"] == sp and r["status"] == "entered"
                       and (r["sym"], r["tf"], r["conf_ts"], r["direction"]) in keys]
                out[v][sp] = metrics([r["tp2"] for r in sub], [r["w"] for r in sub])
        return out
    results["subset_choch_BTC30"] = subset_block(group_rows[("BTC", "30m")])
    results["subset_choch_POOLED30"] = subset_block(pooled30)
    # A on choch subset vs A on no-choch subset (selection effect on baseline entry)
    def a_split(rows):
        out = {}
        for sp in ("cal", "hold"):
            for flag, nm in ((True, "choch"), (False, "nochoch")):
                sub = [r for r in rows if r["variant"] == "A" and r["split"] == sp and r["status"] == "entered" and r["choch"] == flag]
                out[f"{sp}_{nm}"] = metrics([r["tp2"] for r in sub], [r["w"] for r in sub])
        return out
    results["A_selection_BTC30"] = a_split(group_rows[("BTC", "30m")])
    # Q5 subsets (BTC30 own quintile; pooled quintile for pooled)
    def q5_block(rows, qkey):
        out = {}
        for v in "ABC":
            for sp in ("cal", "hold"):
                sub = [r for r in rows if r["variant"] == v and r["split"] == sp and r["status"] == "entered" and r[qkey] == 5]
                out[f"{v}_{sp}"] = metrics([r["tp2"] for r in sub], [r["w"] for r in sub])
        return out
    results["Q5_BTC30"] = q5_block(group_rows[("BTC", "30m")], "quintile")
    results["Q5_POOLED30"] = q5_block(pooled30, "quintile_pooled")
    results["quintile_thresholds"] = {f"{s}_{b}": group_rows[(s, b) + ("qs",)] for (s, b) in
                                      [("BTC", "30m"), ("BTC", "15m"), ("BTC", "1H"), ("ETH", "30m"), ("SOL", "30m"), ("DOGE", "30m")]}
    results["pooled_thresholds"] = pqs.tolist()
    # C variant status breakdown
    cstat = {}
    for r in all_rows:
        if r["variant"] == "C" and r["tf"] == "30m":
            cstat[r["status"]] = cstat.get(r["status"], 0) + 1
    results["C_status_pooled30"] = cstat

    # --- Baseline A reproduction check vs div12m events.csv (BTC 30m) ---
    repro = {}
    if os.path.exists(DIV_EVENTS):
        ref = {}
        with open(DIV_EVENTS) as f:
            rd = csv.DictReader(f)
            for row in rd:
                if row["timeframe"] != "30m": continue
                sp = row["split"]
                if sp not in ("cal", "hold"): continue
                ref.setdefault(sp, []).append(float(row["netR_tp2"]))
        mine = {}
        for sp in ("cal", "hold"):
            sub = [r for r in group_rows[("BTC", "30m")] if r["variant"] == "A" and r["split"] == sp and r["status"] == "entered"]
            mine[sp] = metrics([r["tp2"] for r in sub], [r["w"] for r in sub])
        for sp in ("cal", "hold"):
            rm = metrics(ref.get(sp, []), [1.0] * len(ref.get(sp, [])))
            repro[sp] = dict(div12m=dict(n=rm["n"], avg=rm["avg"], pf=rm["pf"]),
                             mine=dict(n=mine[sp]["n"], avg=mine[sp]["avg"], pf=mine[sp]["pf"]))
    results["repro_A_BTC30"] = repro

    def clean(x):
        if isinstance(x, dict): return {k: clean(v) for k, v in x.items()}
        if isinstance(x, float) and (x == float("inf") or x == float("-inf")): return str(x)
        return x
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(clean(results), f, indent=1, default=str)

    # trades.csv
    cols = ["symbol", "timeframe", "conf_time_utc", "direction", "strength", "quintile", "split",
            "choch", "choch_delay_bars", "level", "pre_sl_touch", "variant", "status",
            "entry_time_utc", "entry", "sl", "R", "cost_R", "netR_tp2", "grossR_tp2",
            "netR_trail03", "netR_ts48", "netR_ts96", "weight"]
    with open(os.path.join(HERE, "trades.csv"), "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(cols)
        for r in sorted(all_rows, key=lambda r: (r["tf"], r["sym"], r["conf_ts"], r["variant"])):
            ct = datetime.fromtimestamp(r["conf_ts"] / 1000, timezone.utc).strftime("%Y-%m-%d %H:%M")
            et = datetime.fromtimestamp(r["entry_ts"] / 1000, timezone.utc).strftime("%Y-%m-%d %H:%M") if r.get("entry_ts") else ""
            wr.writerow([r["sym"], r["tf"], ct, r["direction"], f"{r['strength']:.4f}", r.get("quintile", ""),
                         r["split"], int(bool(r["choch"])), r["choch_delay"] if r["choch_delay"] is not None else "",
                         f"{r['level']:.2f}" if r["level"] is not None else "", r["pre_sl_touch"],
                         r["variant"], r["status"], et,
                         f"{r['entry']:.2f}" if r.get("entry") is not None else "", f"{r['sl']:.2f}",
                         f"{r['R']:.2f}" if "R" in r else "", f"{r['cost_r']:.4f}" if "cost_r" in r else "",
                         f"{r['tp2']:.4f}" if "tp2" in r else "", f"{r['tp2_gross']:.4f}" if "tp2_gross" in r else "",
                         f"{r['trail03']:.4f}" if "trail03" in r else "", f"{r['ts48']:.4f}" if "ts48" in r else "",
                         f"{r['ts96']:.4f}" if "ts96" in r else "", f"{r['w']:.4f}"])
    print(json.dumps(clean(results), indent=1, default=str))

if __name__ == "__main__":
    main()
