"""Combine every available U4 attempt, including partial trajectories, by intervention timing."""
import argparse
import json
from pathlib import Path


GROUPS = {"early": "卸载后早探", "late": "右推 2.5 s",
          "near": "近后墙触发", "weak": "右推 6 s", "sham": "零位移"}
METRICS = {"handoff_weak": "换臂侧载 0.1–1 N", "rear_S5": "右推 S5", "rear_S6": "右推 S6"}


def rate(n, d):
    return n / d if d else None


def counts(entries, metric):
    pos = neg = tp = fp = 0
    for item in entries:
        if metric == "handoff_weak":
            hits, false = item["weak_side"], item["side"]
        else:
            hits = false = item["states"][metric[-2:]]
        pos += hits["true"]
        tp += hits["tp"]
        neg += false["negative"]
        fp += false["fp"]
    return {"runs": len(entries), "positive": pos, "tp": tp, "negative": neg, "fp": fp,
            "recall": rate(tp, pos), "fpr": rate(fp, neg)}


def comparison(first, second, metric, labels):
    a, b = counts(first, metric), counts(second, metric)
    return {labels[0]: a, labels[1]: b, **{
        "delta_"+k+"_pp": (100*(b[k] - a[k]) if b[k] is not None and a[k] is not None else None)
        for k in ("recall", "fpr")}}


def group_results(trials):
    scored = [score for _, score, _ in trials if score is not None]
    episodes = [q["episode"] for q in scored if q["episode"].get("segments")]
    # A runner outcome without paired scores still counts as an attempted trajectory.
    started = sum(bool(q["episode"].get("segments")) if q is not None else bool(phys.get("probe"))
                  for _, q, phys in trials)
    paired, matched = {}, {}
    for metric in METRICS:
        window = "handoff" if metric == "handoff_weak" else "rear_stroke"
        parts = [q["windows"][window] for q in scored if window in q["windows"]]
        paired[metric] = comparison([part["passive"] for part in parts], [part["active"] for part in parts],
                                    metric, ("passive", "active"))
        if metric != "handoff_weak":
            matched_scores = [q for q in scored if q.get("matched_no_pulse", {}).get("common_rear_ticks", 0) > 0]
            matched[metric] = comparison([q["matched_no_pulse"]["common_rear_prefix"] for q in matched_scores],
                                         [q["common_rear_prefix"]["passive"] for q in matched_scores],
                                         metric, ("no_pulse", "pulse_trajectory_passive"))

    known_pre = [e for e in episodes if e.get("pre_end_loaded") is not None]
    clear = [e for e in known_pre if not e["pre_end_loaded"]]
    new_load = [e for e in clear if e.get("single_tick_contact_during_previously_clear")]
    stable = [e for e in new_load if e.get("sustained_contact_during_previously_clear")]
    stats = {"attempts": len(trials), "scored": len(scored), "entered": started, "unreached": len(trials)-started,
             "complete_cycle": sum(e["complete_sequence"] for e in episodes) +
                  sum(bool(p.get("probe", {}).get("end_step")) for _, q, p in trials if q is None and p.get("probe")),
             "complete_rear_scored": sum(q["complete_rear"] for q in scored),
             "seated": sum((q["physical_status"] if q else p.get("status")) == "SEATED_STABLE" for _,q,p in trials),
             "stopped": sum((q["physical_status"] if q else p.get("status")) == "STOPPED" for _,q,p in trials),
             "changed_state_steps": sum(q["changed_state_predictions"] for q in scored),
             "changed_side_steps": sum(q["changed_side_predictions"] for q in scored)}
    weak_pre = [e for e in known_pre if e["pre_end_loaded"] and
                e["segments"]["pre"]["median_side_N"] is not None and .1 <= e["segments"]["pre"]["median_side_N"] < 1.]
    diagnosis = {"pre_known": len(known_pre), "pre_loaded": len(known_pre)-len(clear), "pre_clear": len(clear),
                 "weak_pre": len(weak_pre),
                 "correct_weak_pre": sum(e.get("estimated_pre_loaded") is True for e in weak_pre),
                 "changed_by_pre_hold": sum(e.get("load_changed_during_pre_hold") is True for e in episodes),
                 "pre_reported_on_known": sum(e.get("estimated_pre_loaded") is not None for e in known_pre),
                 "pre_correct": sum(e.get("retrospective_correct") is True for e in known_pre),
                 "pre_false_positive": sum(e.get("estimated_pre_loaded") is True for e in clear),
                 "clear_then_any_load": len(new_load), "clear_then_only_short_load": len(new_load)-len(stable),
                 "clear_then_sustained_load": len(stable),
                 "short_only_events_load_ticks": sum(e["during_load_ticks"] for e in new_load if
                                                     not e.get("sustained_contact_during_previously_clear"))}
    return {"trials": [name for name,_,_ in trials], "outcomes": stats, "same_trajectory": paired,
            "matched_rear_prefix": matched, "probe_events": diagnosis}


def fraction(n, d):
    return f"{n}/{d}" if d else "—"


def fmt(v):
    return f"{v:+.2f}" if v is not None else "—"


def markdown(groups):
    text = ["# 上层第 4 箱侧探分组结果", "",
            "计数单位为独立物理试次或 120 Hz 物理步。已中止和未触发试次按其实际记录计入。未观测到相应真值类别时，比率标为 —。", "",
            "| 时机 | 记录试次 | 有评分 | 已介入 | 未触发 | 全部 192 步 | 后推完整（有评分） | 稳定就位 | 中止 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    def row(*cells):
        text.append("| " + " | ".join(str(v) for v in cells) + " |")
    for key, g in groups.items():
        row(GROUPS[key], *(g["outcomes"][k] for k in
            ("attempts","scored","entered","unreached","complete_cycle","complete_rear_scored","seated","stopped")))
    text.extend(["", "## 同一运动、同一步的主动与被动评分", "",
                 "召回按 TP/P，误报按 FP/N。弱载行的 P 是轴向承载 0.1 ≤ F < 1 N 的步数，N 是明确无侧载的步数。S5/S6 的 N 是已知真值为其他状态的步数。差值为主动减被动，单位为百分点。", "",
                 "| 时机 | 指标 | 有该段试次 | 被动 TP/P | 主动 TP/P | Δ召回 | 被动 FP/N | 主动 FP/N | Δ误报率 |",
                 "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for key, g in groups.items():
        for m, x in g["same_trajectory"].items():
            a, b = x["passive"], x["active"]
            row(GROUPS[key], METRICS[m], a["runs"], fraction(a["tp"],a["positive"]),
                fraction(b["tp"],b["positive"]), fmt(x["delta_recall_pp"]),
                fraction(a["fp"],a["negative"]), fraction(b["fp"],b["negative"]), fmt(x["delta_fpr_pp"]))
    text.extend(["", "## 探测前状态与探测中首次出现的侧载", "",
                 "探前真值取 12 步零位移准备段最后连续三步。此表的探前弱载要求准备段侧载中位数在 [0.1, 1) N，逐帧召回仍按各物理步计数。补充试次选取使用介入前的另一个 12 步窗口，详见方案。短脉冲指从开始侧移到回程结束期间有物理传力，但最长连续不足三步。恢复稳定三步承载记入持续一栏。零位移组按相同时段统计。", "",
                 "| 时机 | 探前受载 / 明确 | 探前已脱开 | 正确 / 明确 | 已作判断 / 明确 | 探前弱载命中 / 实际弱载 | 预报承载误报 / 已脱开 | 脱开后仅短脉冲（次） | 短脉冲承载步 | 脱开后持续（次） |",
                 "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for key, g in groups.items():
        d = g["probe_events"]
        clear = d["pre_clear"]
        row(GROUPS[key], fraction(d["pre_loaded"],d["pre_known"]), clear, fraction(d["pre_correct"],d["pre_known"]),
            fraction(d["pre_reported_on_known"],d["pre_known"]), fraction(d["correct_weak_pre"],d["weak_pre"]),
            fraction(d["pre_false_positive"],clear),
            d["clear_then_only_short_load"] if clear else "—", d["short_only_events_load_ticks"] if clear else "—",
            d["clear_then_sustained_load"] if clear else "—")
    text.extend(["", "## 同偏置的无探测历史运行", "",
                 "两条不同物理轨迹均使用原被动判据。每一对从右推开始取相同数量的步。差值为带干预轨迹减历史无探测轨迹，接触分母可不同。同一时机组内才合计。", "",
                 "| 时机 | 指标 | 可配对试次 | 历史 TP/P | 干预 TP/P | Δ召回 | 历史 FP/N | 干预 FP/N | Δ误报率 |",
                 "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for key, g in groups.items():
        for m,x in g["matched_rear_prefix"].items():
            a,b = x["no_pulse"],x["pulse_trajectory_passive"]
            row(GROUPS[key],METRICS[m],a["runs"],fraction(a["tp"],a["positive"]),fraction(b["tp"],b["positive"]),
                fmt(x["delta_recall_pp"]),fraction(a["fp"],a["negative"]),fraction(b["fp"],b["negative"]),fmt(x["delta_fpr_pp"]))
    text.extend(["", "## 纳入的物理试次", ""])
    for key, g in groups.items():
        if g["trials"]:
            text.append("- " + GROUPS[key] + "　" + "、".join(g["trials"]))
    return "\n".join(text)+"\n"


def summarize(root):
    outcome_file = root/"physical_outcomes.json"
    outcomes = json.loads(outcome_file.read_text()) if outcome_file.exists() else {}
    scores = {p.parent.name: json.loads(p.read_text()) for p in sorted(root.glob("*/scores.json"))}
    cases = {key: [] for key in GROUPS}
    for name in sorted(outcomes.keys() | scores.keys()):
        record = scores.get(name)
        if record is not None and record["source"] != name:
            raise ValueError(f"Directory and score identity disagree: {name}")
        cases[name.split("_",1)[0]].append((name,record,outcomes.get(name,{})))
    results = {k: group_results(v) for k,v in cases.items()}
    (root/"summary.json").write_text(json.dumps(results, indent=2, allow_nan=False)+"\n")
    (root/"summary.md").write_text(markdown(results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    summarize(parser.parse_args().root)
