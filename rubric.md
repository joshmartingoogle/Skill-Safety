# Skill Safety Index Rubric

> **Directionality Convention:** All six scored items (`N1`–`N5` and `S`) are oriented consistently in the **Skill Safety** direction (`0` = Lowest Skill Safety [Highest Unassisted Need / Highest Stakes]; `3` = Highest Skill Safety [Lowest Unassisted Need / Lowest Stakes]). Greater skill safety corresponds to lower unassisted need (`N` items scored higher toward `3`) and lower error stakes (`S` scored higher toward `3`).

## Gate (Asked First)

Can today's AI produce the content of this item, such as a draft, an analysis, a plan, or a check? If so, the item is scored, even when a person must deliver, approve, or check the result. Only items where AI adds nothing of substance—such as hands-on physical work, pure observation, or completing required training in person—are set aside (`ai_assistable = false`).

## Scored Rubric Table (0 = Lowest Skill Safety, 3 = Highest Skill Safety)

Each question is answered `0` to `3` in the **Skill Safety** direction. Answer each on its own, give one sentence of reasoning, and assign a certainty score (`0.0` to `1.0`). `N1` to `N5` cover **unassisted need** (higher score = lower need = greater safety), and `S` covers **stakes** (higher score = lower stakes = greater safety). Erosion is recorded as a separate flag below the table.

| Item | Dimension & Question | 0 (Lowest Safety / Highest Need or Stakes) | 1 (Low Safety / High Need or Stakes) | 2 (Moderate Safety / Low Need or Stakes) | 3 (Highest Safety / Lowest Need or Stakes) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **N1** | **Fallback.** If AI were unexpectedly unavailable when this came up, how long could it wait? | It cannot wait at all. | A few hours. | Until the next day. | Several days or more. |
| **N2** | **Live performance.** How often must the person do this on the spot, in front of others, with no chance to consult AI? Presenting or defending work prepared earlier with AI does not count; that can be rehearsed. | Daily. | About weekly. | About monthly. | Rarely or never. |
| **N3** | **No-AI rules.** How often must the person do this where client, employer, court or confidentiality rules do not allow AI? | Daily. | About weekly. | About monthly. | Rarely or never. |
| **N4** | **Checking.** To tell whether a version produced by AI is right, how much of the work has to be repeated? For a judgment call with no single right answer, such as an estimate or a strategy, ask how much of the reasoning has to be repeated to trust it. | All of it. The work has to be redone. | Much of it. Sources or key steps have to be rechecked. | A little. A careful read is enough. | None. A glance or an automatic test is enough. |
| **N5** | **Foundation.** Is doing this by hand how people in this occupation build other skills they need? Name them. | Yes. Training and entry-level work are built around it. | Mostly. It is the usual way the later skill is learned. | Partly. It is one of several ways the later skill is learned. | No. Nothing later is learned by doing this. |
| **S** | **Stakes.** In the typical case, if the person got this wrong once and no one caught it, what would it cost, counting harm to clients, patients, or the public, and to the person who has to answer for the work? | Serious harm to a person that cannot be undone, or takes years to undo. | Lasting cost to a client or the organisation, such as lost money or a sanction. | Fixed within weeks; others are delayed. | Fixed within a day; no one outside the team is affected. |

## Learning Flag (Yes or No, Not Scored in Index)

* **Learning flag (`yes` or `no`):** *"With today's AI used in the ordinary way, can a person still learning this item get a usable result without working through the reasoning themselves?"*

## Anchor Examples for the Ends of Each Scale (Skill Safety Direction)

* **N1 — Fallback:** `0` (Least Safe) is an airline pilot flying by hand when the autopilot disconnects, which cannot wait at all. `3` (Most Safe) is a marketing analyst drafting a campaign summary, which can wait until the tool is back.
* **N2 — Live performance:** `0` (Least Safe) is a lawyer answering a judge's unexpected question on the law. `3` (Most Safe) is a bookkeeper reconciling accounts.
* **N3 — No-AI rules:** `0` (Least Safe) is a lawyer on a matter where the client has barred AI. `3` (Most Safe) is a marketing analyst drafting a campaign summary.
* **N4 — Checking:** `0` (Least Safe) is a legal research memo, where a missed case shows only by redoing the search (`1` is a translation of a contract, which has to be compared with the original). `3` (Most Safe) is a short piece of code that an automated test either passes or fails.
* **N5 — Foundation:** `0` (Least Safe) is a child doing arithmetic by hand, which builds the number sense that later maths depends on. `3` (Most Safe) is a delivery driver finding a route.
* **S — Stakes:** `0` (Least Safe) is a nurse giving the wrong dose. `3` (Most Safe) is a typo in an internal note, fixed the same day.

## Certainty-Weighted Skill Safety Index Construction

For each item $i$ and rubric dimension $k \in \{N_1, N_2, N_3, N_4, N_5, S\}$, the model produces:
1. A directional safety score $x_{i,k} \in \{0, 1, 2, 3\}$ (where $3$ = safest / lowest need or stakes, and $0$ = least safe / highest need or stakes).
2. An item-dimension certainty score $c_{i,k} \in [0, 1]$ reflecting the epistemic confidence of that assessment.

The **Certainty-Weighted Skill Safety Index** ($I_i \in [0, 3]$, or normalized to $[0, 1]$ by dividing by $3$) for item $i$ is defined as the certainty-weighted mean across the six rubric dimensions:

$$I_i = \frac{\sum_{k \in \{N_1, N_2, N_3, N_4, N_5, S\}} c_{i,k} \cdot x_{i,k}}{\sum_{k \in \{N_1, N_2, N_3, N_4, N_5, S\}} c_{i,k}}$$
