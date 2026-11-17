#!/usr/bin/env python3
"""
Skill Safety Index Scoring Pipeline
-----------------------------------
Individually passes each item across three occupational taxonomies:
  1. Stanford TaskNet (Nov 17, 2026 release; N = 202 tasks across 6 occupations/industries)
  2. O*NET 31.0 Tasks (task_statements.csv; N = 118 tasks across the 6 focal occupations/industries)
  3. O*NET 31.0 Skills (essential_skills.csv + transferable_skills.csv + content_model_reference.csv;
     N = 210 occupation-skill items = 35 skills x 6 occupations/industries)

Each item is scored individually via the Gemini Flash API (targeting gemini-3.8-flash with
automatic fallback on endpoint timeout/503) on the directionally consistent Skill Safety Index
Rubric (0 = Lowest Skill Safety [Highest Unassisted Need / Highest Stakes], 3 = Highest Skill Safety
[Lowest Unassisted Need / Lowest Stakes]), with an explicit certainty score (0.0 to 1.0) for each
assessment made. Uses native structured responseSchema (reasoning before score, no example numbers
in the prompt) to prevent prompt-template numeric anchoring.
"""

import concurrent.futures
import json
import os
import random
import threading
import time
import urllib.error
import urllib.request
from typing import Dict, Any, List
import pandas as pd
import numpy as np

BASE_DIR = "/usr/local/google/home/joshuabmartin/Skills Project"
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)


def load_env_api_key(env_path: str) -> str:
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("GEMINI_API_KEY="):
                    return line.split("=", 1)[1].strip()
    return os.environ.get("GEMINI_API_KEY", "")


API_KEY = load_env_api_key(os.path.join(BASE_DIR, ".env"))
PRIMARY_MODEL = "gemini-3.8-flash"
FALLBACK_MODELS = ["gemini-3-flash-preview", "gemini-3.1-flash-lite"]

SOC_TO_INDUSTRY = {
    "15-1252.00": "Information & Technology",
    "23-1011.00": "Legal & Professional Services",
    "25-2021.00": "Educational Services",
    "29-1214.00": "Healthcare: Clinical Medicine",
    "31-1122.00": "Healthcare: Personal & Home Care",
    "35-9021.00": "Hospitality & Food Services",
}

SOC_TO_OCC_SHORT = {
    "15-1252.00": "Software Developers",
    "23-1011.00": "Lawyers",
    "25-2021.00": "Elementary School Teachers",
    "29-1214.00": "Emergency Medicine Physicians",
    "31-1122.00": "Personal Care Aides",
    "35-9021.00": "Dishwashers",
}

RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "gate_reasoning": {"type": "STRING"},
        "gate_ai_assistable": {"type": "BOOLEAN"},
        "gate_certainty": {"type": "NUMBER"},
        "N1_reasoning": {"type": "STRING"},
        "N1_score": {"type": "INTEGER"},
        "N1_certainty": {"type": "NUMBER"},
        "N2_reasoning": {"type": "STRING"},
        "N2_score": {"type": "INTEGER"},
        "N2_certainty": {"type": "NUMBER"},
        "N3_reasoning": {"type": "STRING"},
        "N3_score": {"type": "INTEGER"},
        "N3_certainty": {"type": "NUMBER"},
        "N4_reasoning": {"type": "STRING"},
        "N4_score": {"type": "INTEGER"},
        "N4_certainty": {"type": "NUMBER"},
        "N5_reasoning": {"type": "STRING"},
        "N5_score": {"type": "INTEGER"},
        "N5_certainty": {"type": "NUMBER"},
        "S_reasoning": {"type": "STRING"},
        "S_score": {"type": "INTEGER"},
        "S_certainty": {"type": "NUMBER"},
        "learning_flag": {"type": "BOOLEAN"},
        "learning_flag_certainty": {"type": "NUMBER"},
    },
    "required": [
        "gate_reasoning", "gate_ai_assistable", "gate_certainty",
        "N1_reasoning", "N1_score", "N1_certainty",
        "N2_reasoning", "N2_score", "N2_certainty",
        "N3_reasoning", "N3_score", "N3_certainty",
        "N4_reasoning", "N4_score", "N4_certainty",
        "N5_reasoning", "N5_score", "N5_certainty",
        "S_reasoning", "S_score", "S_certainty",
        "learning_flag", "learning_flag_certainty",
    ],
}

RUBRIC_PROMPT_TEMPLATE = """You are an expert labor economist and occupational analyst evaluating a single occupational task or skill using the **Skill Safety Index Rubric**.

CRITICAL DIRECTIONALITY RULE:
Every scored question (`N1`, `N2`, `N3`, `N4`, `N5`, and `S`) is oriented in the **Skill Safety** direction:
- **Higher score (3 = Highest Skill Safety)** means **LOWER unassisted need** (`N1`–`N5`) and **LOWER error stakes** (`S`), i.e., it is SAFER to rely on AI assistance permanently without maintaining unassisted human ability.
- **Lower score (0 = Lowest Skill Safety)** means **HIGHER unassisted need** (`N1`–`N5`) and **HIGHER error stakes** (`S`), i.e., humans MUST build and retain the unassisted skill.
Use the full 0–3 scale across items as appropriate to the specific task and occupation.

### Item to Evaluate (Evaluate THIS single item in the context of its occupation and industry):
- **Taxonomy**: {taxonomy}
- **Item ID**: {item_id}
- **Occupation (SOC)**: {occupation} ({soc})
- **Industry Sector**: {industry}
- **Item Statement / Skill Description**: "{statement}"

### Step 1: Gate Question (Asked First)
Can today's AI produce the content of this item, such as a draft, an analysis, a plan, or a check? If so, `gate_ai_assistable` is `true`, even when a person must deliver, approve, or check the result. Only items where AI adds nothing of substance—such as hands-on physical work, pure observation, or completing required training in person—have `gate_ai_assistable` set to `false`.
*(Note: Even if `gate_ai_assistable` is `false`, still provide scores `0`–`3` and certainties `0.0`–`1.0` for `N1`–`N5` and `S` evaluating the task/skill itself in this occupation).*

### Step 2: Six Scored Skill Safety Dimensions (0 = Lowest Skill Safety, 3 = Highest Skill Safety)
For each dimension, write one sentence of reasoning (`*_reasoning`), assign an integer score `0`, `1`, `2`, or `3` (`*_score`), and assign a calibrated epistemic certainty score in `[0.0, 1.0]` (`*_certainty`) reflecting how unambiguous the rating is for this specific item:

1. **N1 (Fallback Safety)**: *If AI were unexpectedly unavailable when this came up, how long could it wait?*
   - `0` = It cannot wait at all. (Highest fallback need / Lowest safety; e.g., airline pilot flying by hand when autopilot disconnects)
   - `1` = A few hours.
   - `2` = Until the next day.
   - `3` = Several days or more. (Lowest fallback need / Highest safety; e.g., marketing analyst drafting a campaign summary)

2. **N2 (Live Performance Safety)**: *How often must the person do this on the spot, in front of others, with no chance to consult AI? (Presenting or defending work prepared earlier with AI does not count; that can be rehearsed.)*
   - `0` = Daily. (Highest live performance need / Lowest safety; e.g., lawyer answering a judge's unexpected question)
   - `1` = About weekly.
   - `2` = About monthly.
   - `3` = Rarely or never. (Lowest live performance need / Highest safety; e.g., bookkeeper reconciling accounts)

3. **N3 (No-AI Rules Safety)**: *How often must the person do this where client, employer, court, or confidentiality rules do not allow AI?*
   - `0` = Daily. (Highest regulatory/policy restriction / Lowest safety)
   - `1` = About weekly.
   - `2` = About monthly.
   - `3` = Rarely or never. (Lowest restriction / Highest safety)

4. **N4 (Checking Safety)**: *To tell whether a version produced by AI is right, how much of the work has to be repeated? (For a judgment call with no single right answer, ask how much of the reasoning has to be repeated to trust it.)*
   - `0` = All of it. The work has to be redone. (Highest checking need / Lowest safety; e.g., legal research memo where a missed case shows only by redoing the search)
   - `1` = Much of it. Sources or key steps have to be rechecked. (e.g., contract translation compared against original)
   - `2` = A little. A careful read is enough.
   - `3` = None. A glance or an automatic test is enough. (Lowest checking need / Highest safety; e.g., code verified by automated unit test)

5. **N5 (Foundation Safety)**: *Is doing this by hand how people in this occupation build other skills they need?*
   - `0` = Yes. Training and entry-level work are built around it. (Highest foundational need / Lowest safety; e.g., arithmetic by hand building number sense)
   - `1` = Mostly. It is the usual way the later skill is learned.
   - `2` = Partly. It is one of several ways the later skill is learned.
   - `3` = No. Nothing later is learned by doing this. (Lowest foundational need / Highest safety; e.g., delivery driver finding a route)

6. **S (Stakes Safety)**: *In the typical case, if the person got this wrong once and no one caught it, what would it cost, counting harm to clients, patients, or the public, and to the person who has to answer for the work?*
   - `0` = Serious harm to a person that cannot be undone, or takes years to undo. (Highest stakes / Lowest safety; e.g., nurse giving wrong dose)
   - `1` = Lasting cost to a client or the organisation, such as lost money or a sanction.
   - `2` = Fixed within weeks; others are delayed.
   - `3` = Fixed within a day; no one outside the team is affected. (Lowest stakes / Highest safety; e.g., typo in an internal note)

### Step 3: Learning Flag (Boolean, Not Scored in Index)
- **learning_flag**: *"With today's AI used in the ordinary way, can a person still learning this item get a usable result without working through the reasoning themselves?"* (`true` or `false`, plus `learning_flag_certainty` in `[0.0, 1.0]`).
"""


def build_taxonomy_items() -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []

    # 1. Stanford TaskNet (Nov 17, 2026 release; 202 tasks)
    tn_df = pd.read_csv(os.path.join(DATA_DIR, "tasknet-data-11-17.csv"))
    for _, row in tn_df.iterrows():
        soc = str(row["soc"]).strip()
        occ = SOC_TO_OCC_SHORT.get(soc, str(row["occupation"]).strip())
        ind = SOC_TO_INDUSTRY.get(soc, "Other Services")
        items.append({
            "taxonomy": "Stanford TaskNet",
            "item_id": f"TN_{row['id']}",
            "raw_id": str(row["id"]),
            "soc": soc,
            "occupation": occ,
            "industry": ind,
            "statement": str(row["statement"]).strip(),
            "status": str(row.get("status", "core")),
            "relevance": float(row["relevance"]) if pd.notna(row.get("relevance")) else 1.0,
            "importance": float(row["importance"]) if pd.notna(row.get("importance")) else np.nan,
        })

    # 2. O*NET 31.0 Tasks (task_statements.csv across the 6 focal occupations; 118 tasks)
    ts_df = pd.read_csv(os.path.join(DATA_DIR, "task_statements.csv"))
    ts_sub = ts_df[ts_df["O*NET-SOC Code"].isin(SOC_TO_INDUSTRY.keys())].copy()
    for _, row in ts_sub.iterrows():
        soc = str(row["O*NET-SOC Code"]).strip()
        occ = SOC_TO_OCC_SHORT.get(soc, str(row["Title"]).strip())
        ind = SOC_TO_INDUSTRY.get(soc, "Other Services")
        tid = str(row["Task ID"]).strip()
        items.append({
            "taxonomy": "O*NET Tasks",
            "item_id": f"ONET_TASK_{soc}_{tid}",
            "raw_id": tid,
            "soc": soc,
            "occupation": occ,
            "industry": ind,
            "statement": str(row["Task"]).strip(),
            "status": str(row.get("Task Type", "Core")).lower(),
            "relevance": 1.0,
            "importance": np.nan,
        })

    # 3. O*NET 31.0 Skills (essential_skills.csv + transferable_skills.csv + content_model_reference.csv; 35 skills x 6 SOCs = 210 items)
    es_df = pd.read_csv(os.path.join(DATA_DIR, "essential_skills.csv"))
    tr_df = pd.read_csv(os.path.join(DATA_DIR, "transferable_skills.csv"))
    cm_df = pd.read_csv(os.path.join(DATA_DIR, "content_model_reference.csv"))
    skills_df = pd.concat([es_df, tr_df], ignore_index=True)
    skills_im = skills_df[
        (skills_df["Scale ID"] == "IM") & (skills_df["O*NET-SOC Code"].isin(SOC_TO_INDUSTRY.keys()))
    ].merge(cm_df[["Element ID", "Description"]], on="Element ID", how="left")

    for _, row in skills_im.iterrows():
        soc = str(row["O*NET-SOC Code"]).strip()
        occ = SOC_TO_OCC_SHORT.get(soc, str(row["Title"]).strip())
        ind = SOC_TO_INDUSTRY.get(soc, "Other Services")
        el_id = str(row["Element ID"]).strip()
        el_name = str(row["Element Name"]).strip()
        desc = str(row["Description"]).strip()
        not_rel = str(row.get("Not Relevant", "N")).strip()
        stmt = f"{el_name} ({el_id}): {desc} [Applied to occupational activities of {occ}]"
        items.append({
            "taxonomy": "O*NET Skills",
            "item_id": f"ONET_SKILL_{soc}_{el_id}",
            "raw_id": el_id,
            "soc": soc,
            "occupation": occ,
            "industry": ind,
            "statement": stmt,
            "status": "essential" if el_id.startswith("2.A") else "transferable",
            "relevance": 0.0 if not_rel == "Y" else 1.0,
            "importance": float(row["Data Value"]) if pd.notna(row.get("Data Value")) else np.nan,
        })

    return items


_primary_down_until = 0.0
_lock = threading.Lock()


def score_single_item(item: Dict[str, Any], pass_label: str = "main", temp: float = 0.0) -> Dict[str, Any]:
    """Individually scores a single taxonomy item via Gemini Flash API using structured schema."""
    global _primary_down_until
    prompt = RUBRIC_PROMPT_TEMPLATE.format(**item)

    now = time.time()
    if now < _primary_down_until:
        model_sequence = FALLBACK_MODELS + [PRIMARY_MODEL]
    else:
        model_sequence = [PRIMARY_MODEL] + FALLBACK_MODELS

    last_err = None
    for attempt in range(6):
        model = model_sequence[attempt % len(model_sequence)]
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={API_KEY}"
        gen_cfg: Dict[str, Any] = {
            "temperature": temp,
            "responseMimeType": "application/json",
            "responseSchema": RESPONSE_SCHEMA,
        }
        if model == "gemini-3.8-flash":
            gen_cfg["thinkingConfig"] = {"thinkingLevel": "LOW"}
            timeout_sec = 3.5
        elif model == "gemini-3-flash-preview":
            gen_cfg["thinkingConfig"] = {"thinkingLevel": "MINIMAL"}
            timeout_sec = 16.0
        else:
            gen_cfg["thinkingConfig"] = {"thinkingBudget": 0}
            timeout_sec = 15.0

        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": gen_cfg,
        }
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            raw = urllib.request.urlopen(req, timeout=timeout_sec).read().decode("utf-8")
            body = json.loads(raw)
            txt = body["candidates"][0]["content"]["parts"][0]["text"]
            parsed = json.loads(txt)

            dims = ["N1", "N2", "N3", "N4", "N5", "S"]
            out_row = dict(item)
            out_row["pass_label"] = pass_label
            out_row["model_used"] = model
            out_row["gate_ai_assistable"] = bool(parsed.get("gate_ai_assistable", True))
            out_row["gate_certainty"] = float(np.clip(float(parsed.get("gate_certainty", 0.85)), 0.0, 1.0))
            out_row["gate_reasoning"] = str(parsed.get("gate_reasoning", "")).strip()
            out_row["learning_flag"] = bool(parsed.get("learning_flag", False))
            out_row["learning_flag_certainty"] = float(np.clip(float(parsed.get("learning_flag_certainty", 0.80)), 0.0, 1.0))

            scores = []
            certs = []
            for d in dims:
                sc = int(np.clip(round(float(parsed.get(f"{d}_score", 1))), 0, 3))
                ct = float(np.clip(float(parsed.get(f"{d}_certainty", 0.85)), 0.01, 1.0))
                rs = str(parsed.get(f"{d}_reasoning", "")).strip()
                out_row[f"{d}_score"] = sc
                out_row[f"{d}_certainty"] = ct
                out_row[f"{d}_reasoning"] = rs
                scores.append(sc)
                certs.append(ct)

            scores_arr = np.array(scores, dtype=float)
            certs_arr = np.array(certs, dtype=float)

            cw_index = float(np.sum(scores_arr * certs_arr) / np.sum(certs_arr))
            uw_index = float(np.mean(scores_arr))
            cw_need = float(np.sum(scores_arr[:5] * certs_arr[:5]) / np.sum(certs_arr[:5]))
            mean_cert = float(np.mean(certs_arr))

            out_row["safety_index_cw"] = round(cw_index, 1)
            out_row["safety_index_cw_norm"] = round(cw_index / 3.0, 1)
            out_row["safety_index_uw"] = round(uw_index, 1)
            out_row["need_safety_cw"] = round(cw_need, 1)
            out_row["stakes_safety"] = int(scores_arr[5])
            out_row["mean_certainty"] = round(mean_cert, 1)
            return out_row

        except Exception as e:
            last_err = e
            if model == PRIMARY_MODEL:
                with _lock:
                    _primary_down_until = time.time() + 120.0
            else:
                time.sleep(0.4 * (attempt + 1))

    raise RuntimeError(f"Failed to score item {item['item_id']}: {last_err}")


def main():
    items = build_taxonomy_items()
    print(f"Total items across all 3 taxonomies: {len(items)}")
    for tax, grp in pd.DataFrame(items).groupby("taxonomy"):
        print(f"  - {tax}: {len(grp)} items")

    out_jsonl = os.path.join(OUT_DIR, "scored_items_main.jsonl")
    done_map: Dict[str, Dict[str, Any]] = {}
    if os.path.exists(out_jsonl):
        with open(out_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    done_map[obj["item_id"]] = obj
        print(f"Loaded {len(done_map)} already-scored items from {out_jsonl}")

    todo = [it for it in items if it["item_id"] not in done_map]
    print(f"Remaining items to score individually: {len(todo)}")

    write_lock = threading.Lock()
    t0 = time.time()
    completed = 0

    if todo:
        with open(out_jsonl, "a", encoding="utf-8") as fout:
            with concurrent.futures.ThreadPoolExecutor(max_workers=12) as ex:
                fut_map = {ex.submit(score_single_item, it, "main", 0.0): it["item_id"] for it in todo}
                for fut in concurrent.futures.as_completed(fut_map):
                    iid = fut_map[fut]
                    res = fut.result()
                    with write_lock:
                        fout.write(json.dumps(res) + "\n")
                        fout.flush()
                        done_map[iid] = res
                        completed += 1
                        if completed % 25 == 0 or completed == len(todo):
                            elapsed = time.time() - t0
                            rate = completed / max(elapsed, 1e-6)
                            print(f"  [{completed}/{len(todo)}] scored ({rate:.2f} items/s, elapsed {elapsed:.1f}s)", flush=True)

    all_rows = [done_map[it["item_id"]] for it in items if it["item_id"] in done_map]
    df_out = pd.DataFrame(all_rows)
    csv_path = os.path.join(OUT_DIR, "scored_items_all_taxonomies.csv")
    df_out.to_csv(csv_path, index=False)
    print(f"Saved {len(df_out)} scored items to {csv_path}")

    # Step 5: Spot Verification on a Stratified Random Subsample (15 items per taxonomy = 45 items total)
    spot_jsonl = os.path.join(OUT_DIR, "spot_verification_subsample.jsonl")
    spot_done: Dict[str, Dict[str, Any]] = {}
    if os.path.exists(spot_jsonl):
        with open(spot_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    spot_done[obj["item_id"]] = obj

    rng = random.Random(20261117)
    subsample: List[Dict[str, Any]] = []
    for tax in ["Stanford TaskNet", "O*NET Tasks", "O*NET Skills"]:
        tax_items = [it for it in items if it["taxonomy"] == tax]
        picked = rng.sample(tax_items, min(15, len(tax_items)))
        subsample.extend(picked)

    spot_todo = [it for it in subsample if it["item_id"] not in spot_done]
    print(f"Running independent spot-verification pass on {len(subsample)} random subsample items ({len(spot_todo)} remaining)...")
    if spot_todo:
        with open(spot_jsonl, "a", encoding="utf-8") as fspot:
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as ex:
                fut_map = {ex.submit(score_single_item, it, "spot_verification", 0.1): it["item_id"] for it in spot_todo}
                for fut in concurrent.futures.as_completed(fut_map):
                    res = fut.result()
                    with write_lock:
                        fspot.write(json.dumps(res) + "\n")
                        fspot.flush()
                        spot_done[res["item_id"]] = res

    spot_rows = [spot_done[it["item_id"]] for it in subsample if it["item_id"] in spot_done]
    df_spot = pd.DataFrame(spot_rows)
    df_spot.to_csv(os.path.join(OUT_DIR, "spot_verification_subsample.csv"), index=False)
    print(f"Spot verification complete ({len(df_spot)} items saved).")


if __name__ == "__main__":
    main()
