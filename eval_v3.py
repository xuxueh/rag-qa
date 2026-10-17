"""
eval_v3.py — 100 条核验测试集 · 三种检索方案对比（检索侧评测，不调用大模型）
==========================================================================
对比方案：
  A 纯向量检索（dense）        ：只用 Chroma 语义向量，取 top-10
  B 混合检索（hybrid）         ：向量 + BM25，RRF 融合，取 top-10
  C 混合检索 + Reranker        ：B 的 10 条候选交给 bge-reranker-v2-m3 重排

指标（全部按「期望文档是否出现在第 K 位」计算）：
  Recall@1/3/5/10、MRR（文件级：命中任一期望文档；条款级：文件名 + 条款号都对上）
  AllSource@10（多文档题：期望的所有文档是否都进 top-10）
  每方案平均 / p50 / p95 单次检索耗时
  失败案例清单（文件级排名 > 3 或未命中）

无答案题不参与 Recall 统计，单独分析：最高精排分能不能把「无答案」和「有答案」分开。

用法：
  python eval_v3.py                # 全量（含 reranker，约 2-4 分钟）
  python eval_v3.py --no-rerank    # 只跑 A/B（不需要 reranker 模型）
  python eval_v3.py --limit 20     # 冒烟跑前 20 条
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
import rag_qa
from hybrid_retriever import build_hybrid

log = config.setup_logging()

KS = (1, 3, 5, 10)
EVAL_SET = Path(__file__).with_name("eval_set_v3.json")
OUT_JSON = Path(__file__).with_name("eval_results_v3.json")


# ═══════════════════ 命中判定 ═══════════════════
def expected_sources(item: dict) -> list[str]:
    return [item["source"]] + list(item.get("also") or [])


def file_rank(metas, item: dict) -> int | None:
    """命中任一期望文档的排名（1-based）"""
    want = set(expected_sources(item))
    for i, m in enumerate(metas):
        if m.source in want:
            return i + 1
    return None


def clause_rank(metas, item: dict) -> int | None:
    """条款级：文件名对 + 条款号有交集（只对主文档判定，避免 also 干扰）"""
    arts = set(item.get("articles") or [])
    if not arts or not item.get("source"):
        return None
    for i, m in enumerate(metas):
        if m.source == item["source"] and arts & set(m.articles):
            return i + 1
    return None


def all_sources_in_topk(metas, item: dict) -> bool | None:
    """多文档题：期望的每个文档是否都在候选里"""
    want = set(expected_sources(item))
    if len(want) <= 1:
        return None
    got = {m.source for m in metas}
    return want.issubset(got)


def summarize_ranks(ranks: list[int | None]) -> dict:
    n = len(ranks)
    out = {f"R@{k}": round(sum(1 for r in ranks if r and r <= k) / n, 4) for k in KS}
    out["MRR"] = round(sum(1.0 / r for r in ranks if r) / n, 4)
    out["n"] = n
    return out


def latency_stats(samples_ms: list[float]) -> dict:
    ordered = sorted(samples_ms)
    p = lambda q: ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))]
    return {"mean_ms": round(statistics.mean(ordered), 1),
            "p50_ms": round(p(0.50), 1), "p95_ms": round(p(0.95), 1),
            "max_ms": round(ordered[-1], 1)}


# ═══════════════════ 三种方案 ═══════════════════
def make_modes(retriever, use_rerank: bool):
    from rerank import rerank_objects

    def dense(query):
        return retriever.dense_meta(query, top_k=config.RECALL_TOP_K)

    def hybrid(query):
        return retriever.retrieve_meta(query, top_k=config.RECALL_TOP_K)

    def hybrid_rerank(query):
        recall = retriever.retrieve_meta(query, top_k=config.RECALL_TOP_K)
        ranked, meta = rerank_objects(query, recall, lambda m: m.text, top_n=config.RECALL_TOP_K)
        if meta.get("fallback"):
            raise RuntimeError("reranker 不可用（降级返回原顺序），本次结果不可用于对比")
        return ranked, meta.get("scores") or []

    modes = {
        "A_纯向量": (dense, False),
        "B_混合检索RRF": (hybrid, False),
        "C_混合+Reranker": (hybrid_rerank, True),
    }
    return {k: v for k, v in modes.items() if use_rerank or not v[1]}


# ═══════════════════ 主流程 ═══════════════════
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-rerank", action="store_true", help="跳过 reranker（只跑 A/B）")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条（冒烟）")
    args = parser.parse_args()

    config.assert_valid(require_api_key=False)
    items = json.loads(EVAL_SET.read_text(encoding="utf-8"))
    if args.limit:
        items = items[: args.limit]
    answerable = [i for i in items if i.get("source")]
    unanswerable = [i for i in items if not i.get("source")]
    log.info("测试集：%d 条（有答案 %d / 无答案 %d），分类：%s",
             len(items), len(answerable), len(unanswerable),
             dict(Counter(i["category"] for i in items)))

    log.info("构建知识库...")
    retriever = build_hybrid()
    log.info("chunk 数：%d，文档数：%d", len(retriever.chunks),
             len({c.source for c in retriever.chunks}))

    modes = make_modes(retriever, use_rerank=not args.no_rerank)
    results: dict = {
        "meta": {
            "eval_set": EVAL_SET.name,
            "total": len(items),
            "answerable": len(answerable),
            "unanswerable": len(unanswerable),
            "categories": dict(Counter(i["category"] for i in items)),
            "chunks": len(retriever.chunks),
            "documents": len({c.source for c in retriever.chunks}),
            "config": {"chunk_size": config.CHUNK_SIZE, "chunk_overlap": config.CHUNK_OVERLAP,
                       "recall_top_k": config.RECALL_TOP_K, "rrf_k": config.RRF_K,
                       "bm25_top_k": config.BM25_TOP_K or config.RECALL_TOP_K,
                       "rerank_model": config.RERANK_MODEL,
                       "embedding_model": config.EMBEDDING_MODEL},
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "modes": {},
    }

    # reranker 首次加载不计入单次耗时：先热身一次
    if "C_混合+Reranker" in modes:
        log.info("reranker 预热...")
        modes["C_混合+Reranker"][0](answerable[0]["question"] if answerable else "测试")

    no_answer_scores: dict[str, list[float]] = defaultdict(list)

    for mode_name, (fn, is_rerank) in modes.items():
        log.info("=== 运行方案 %s ===", mode_name)
        file_ranks, clause_ranks, all_hits = [], [], []
        latencies, per_cat = [], defaultdict(lambda: {"file": [], "clause": []})
        failures = []

        for item in items:
            if not item.get("source"):        # 无答案题：只取分数，不进 Recall 统计
                if is_rerank:
                    t0 = time.perf_counter()
                    _, scores = fn(item["question"])
                    latencies.append((time.perf_counter() - t0) * 1000)
                    no_answer_scores["no_answer"].append(max(scores) if scores else -99.0)
                continue

            t0 = time.perf_counter()
            out = fn(item["question"])
            metas = out[0] if isinstance(out, tuple) else out
            elapsed = (time.perf_counter() - t0) * 1000
            latencies.append(elapsed)

            fr = file_rank(metas, item)
            cr = clause_rank(metas, item)
            ah = all_sources_in_topk(metas, item)
            file_ranks.append(fr)
            clause_ranks.append(cr)
            if ah is not None:
                all_hits.append(ah)
            per_cat[item["category"]]["file"].append(fr)
            per_cat[item["category"]]["clause"].append(cr)
            if is_rerank and isinstance(out, tuple) and out[1]:
                no_answer_scores["answerable"].append(max(out[1]))

            if fr is None or fr > 3:
                failures.append({
                    "id": item["id"], "category": item["category"],
                    "question": item["question"],
                    "expected": expected_sources(item),
                    "expected_articles": item.get("articles"),
                    "gold_answer": item.get("answer"),
                    "file_rank": fr, "clause_rank": cr,
                    "got_top5": [{"rank": i + 1, "source": m.source,
                                  "articles": m.articles, "text": m.text[:60]}
                                 for i, m in enumerate(metas[:5])],
                })

        results["modes"][mode_name] = {
            "file_level": summarize_ranks(file_ranks),
            "clause_level": summarize_ranks(clause_ranks),
            "all_sources_in_top10": (round(sum(1 for x in all_hits if x) / len(all_hits), 4)
                                     if all_hits else None),
            "all_sources_n": len(all_hits),
            "latency": latency_stats(latencies),
            "by_category": {cat: {"file": summarize_ranks(v["file"]),
                                  "clause": summarize_ranks(v["clause"])}
                            for cat, v in per_cat.items()},
            "failures": failures,
        }
        log.info("%s → 文件级 R@1 %.1f%% / R@3 %.1f%% / MRR %.3f；平均 %.0fms",
                 mode_name,
                 results["modes"][mode_name]["file_level"]["R@1"] * 100,
                 results["modes"][mode_name]["file_level"]["R@3"] * 100,
                 results["modes"][mode_name]["file_level"]["MRR"],
                 results["modes"][mode_name]["latency"]["mean_ms"])

    # ── 无答案可分性分析（用 C 方案的最高精排分） ──
    if "C_混合+Reranker" in modes and unanswerable and answerable:
        ans = no_answer_scores.get("answerable", [])
        no = no_answer_scores.get("no_answer", [])
        if ans and no:
            best = None
            lo, hi = min(ans + no), max(ans + no)
            for step in range(0, 201):
                th = lo + (hi - lo) * step / 200
                tp = sum(1 for s in no if s < th)      # 无答案且判为无答案
                tn = sum(1 for s in ans if s >= th)    # 有答案且判为有答案
                acc = (tp + tn) / (len(no) + len(ans))
                if best is None or acc > best["accuracy"]:
                    best = {"threshold": round(th, 3), "accuracy": round(acc, 4),
                            "abstain_rate_on_no_answer": round(tp / len(no), 4),
                            "answer_rate_on_answerable": round(tn / len(ans), 4)}
            results["no_answer_analysis"] = {
                "method": "取 C 方案召回池内最高 rerank 分数，低于阈值判为「无答案」",
                "answerable_score": {"min": round(min(ans), 3),
                                     "median": round(statistics.median(ans), 3),
                                     "max": round(max(ans), 3)},
                "no_answer_score": {"min": round(min(no), 3),
                                    "median": round(statistics.median(no), 3),
                                    "max": round(max(no), 3)},
                "best_threshold": best,
                "note": "排序分数不是校准过的概率，仅作「能否用分数设阈值拒答」的可行性参考",
            }

    OUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print_tables(results)
    print(f"\n结果已写入：{OUT_JSON}")
    return 0


def print_tables(results: dict) -> None:
    def row(name, s, lat):
        return (f"| {name} | {s['n']} | {s['R@1']:.1%} | {s['R@3']:.1%} | {s['R@5']:.1%} | "
                f"{s['R@10']:.1%} | {s['MRR']:.3f} | {lat['mean_ms']:.0f} | {lat['p95_ms']:.0f} |")

    print("\n## 文件级检索指标（命中任一期望文档）\n")
    print("| 方案 | n | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR | 平均耗时ms | p95耗时ms |")
    print("|---|---|---|---|---|---|---|---|---|")
    for name, r in results["modes"].items():
        print(row(name, r["file_level"], r["latency"]))

    print("\n## 条款级检索指标（文件名 + 条款号都对上）\n")
    print("| 方案 | n | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR |")
    print("|---|---|---|---|---|---|---|")
    for name, r in results["modes"].items():
        s = r["clause_level"]
        print(f"| {name} | {s['n']} | {s['R@1']:.1%} | {s['R@3']:.1%} | {s['R@5']:.1%} | "
              f"{s['R@10']:.1%} | {s['MRR']:.3f} |")

    print("\n## 分类型文件级 Recall@1（看哪类问题最容易翻车）\n")
    cats = list(next(iter(results["modes"].values()))["by_category"].keys())
    print("| 类型 | " + " | ".join(results["modes"].keys()) + " |")
    print("|---" * (len(results["modes"]) + 1) + "|")
    for cat in cats:
        cells = []
        for r in results["modes"].values():
            c = r["by_category"].get(cat)
            cells.append(f"{c['file']['R@1']:.1%}" if c else "-")
        print(f"| {cat} | " + " | ".join(cells) + " |")

    no = results.get("no_answer_analysis")
    if no:
        print("\n## 无答案题：能不能用精排分数设阈值拒答\n")
        print(f"- 有答案题最高分：min {no['answerable_score']['min']} / "
              f"median {no['answerable_score']['median']} / max {no['answerable_score']['max']}")
        print(f"- 无答案题最高分：min {no['no_answer_score']['min']} / "
              f"median {no['no_answer_score']['median']} / max {no['no_answer_score']['max']}")
        b = no["best_threshold"]
        print(f"- 最佳阈值 {b['threshold']}：准确率 {b['accuracy']:.1%}，"
              f"无答案题拒答率 {b['abstain_rate_on_no_answer']:.1%}，"
              f"有答案题答出率 {b['answer_rate_on_answerable']:.1%}")

    print("\n## 失败案例（文件级排名 > 3 或未命中）\n")
    for name, r in results["modes"].items():
        print(f"### {name}：{len(r['failures'])} 条")
        for f in r["failures"][:8]:
            print(f"- [{f['id']}/{f['category']}] {f['question']}")
            print(f"  - 期望：{f['expected']} 第{f['expected_articles']}条；"
                  f"实际排名 {f['file_rank']}，条款排名 {f['clause_rank']}")
            print(f"  - 实际 top3：{[(g['source'], g['articles']) for g in f['got_top5'][:3]]}")


if __name__ == "__main__":
    raise SystemExit(main())
