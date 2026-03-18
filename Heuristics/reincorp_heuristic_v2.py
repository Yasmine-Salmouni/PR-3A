# -*- coding: utf-8 -*-
"""
reincorp_heuristic_tabu_improved.py

Version améliorée orientée performance pour la recherche tabou.

Principales idées:
- même API de base que le fichier initial
- Tabu multi-voisinage pur: replace / swap / relocation
- pas de ruin&recreate dans Tabu (paramètres legacy acceptés mais ignorés)
- cache LRU borné des évaluations
- candidate lists pré-calculées par slot
- relocation accélérée via slots compatibles par chute
- mémoire de fréquence pour diversifier
- reactive tabu tenure optionnelle
- VND léger / intensification après stagnation
- redémarrage léger contrôlé depuis la meilleure solution
"""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np


# ==========================================================
# Structures
# ==========================================================

@dataclass
class Batch:
    dmc: pd.Timestamp
    dlc: pd.Timestamp
    qty: float


@dataclass(frozen=True)
class Slot:
    date: pd.Timestamp
    produit: str


# ==========================================================
# Chargement des données
# ==========================================================

def load_data(excel_path: str):
    stock_df = pd.read_excel(excel_path, sheet_name="Stock_proj_chute")
    reinc_df = pd.read_excel(excel_path, sheet_name="Reincorpo_max")
    params_df = pd.read_excel(excel_path, sheet_name="Parametres")

    stock_df["DMC"] = pd.to_datetime(stock_df["DMC"])
    stock_df["DLC"] = pd.to_datetime(stock_df["DLC"])
    reinc_df["Date"] = pd.to_datetime(reinc_df["Date"])

    if "Valeur" not in params_df.columns:
        raise ValueError("Onglet Parametres: colonne 'Valeur' introuvable.")
    thr = pd.to_numeric(params_df["Valeur"], errors="coerce").dropna()
    if thr.empty:
        raise ValueError("Onglet Parametres: aucune valeur numérique trouvée dans 'Valeur'.")
    threshold = float(thr.iloc[0])

    batches_by_chute: Dict[str, List[Batch]] = {}
    for row in stock_df.itertuples(index=False):
        c = str(row.Chute)
        batches_by_chute.setdefault(c, []).append(
            Batch(pd.Timestamp(row.DMC), pd.Timestamp(row.DLC), float(row.Quantite))
        )
    for c in batches_by_chute:
        batches_by_chute[c].sort(key=lambda b: b.dlc)

    cap_by_slot: Dict[Slot, Dict[str, float]] = {}
    for row in reinc_df.itertuples(index=False):
        slot = Slot(pd.Timestamp(row.Date), str(row.Produit))
        cap_by_slot.setdefault(slot, {})[str(row.Chute)] = float(row.Quantite)

    slots = sorted(cap_by_slot.keys(), key=lambda s: (s.date, s.produit))
    return threshold, batches_by_chute, cap_by_slot, slots


# ==========================================================
# Outils stock
# ==========================================================

def deepcopy_batches(batches_by_chute: Dict[str, List[Batch]]) -> Dict[str, List[Batch]]:
    return {c: [Batch(b.dmc, b.dlc, b.qty) for b in lst] for c, lst in batches_by_chute.items()}


def available_qty(batches: List[Batch], date: pd.Timestamp) -> float:
    return sum(b.qty for b in batches if b.qty > 0 and b.dmc <= date <= b.dlc)


def earliest_dlc_days(batches: List[Batch], date: pd.Timestamp) -> Optional[int]:
    days = [(b.dlc - date).days for b in batches if b.qty > 0 and b.dmc <= date <= b.dlc]
    return min(days) if days else None


def consume(batches: List[Batch], date: pd.Timestamp, qty: float) -> float:
    remaining = float(qty)
    for b in batches:
        if remaining <= 1e-12:
            break
        if b.qty > 0 and b.dmc <= date <= b.dlc:
            take = min(b.qty, remaining)
            b.qty -= take
            remaining -= take
    return qty - remaining


# ==========================================================
# Évaluation (simulation)
# ==========================================================

def evaluate_solution(
    assignment: Dict[Slot, Optional[str]],
    threshold: float,
    batches_by_chute: Dict[str, List[Batch]],
    cap_by_slot: Dict[Slot, Dict[str, float]],
    slots: List[Slot],
    lambda_campaign: float = 1.0
) -> Tuple[float, int, float, List[Tuple[pd.Timestamp, str, Optional[str], float]]]:
    batches = deepcopy_batches(batches_by_chute)
    total = 0.0
    n_camp = 0
    plan: List[Tuple[pd.Timestamp, str, Optional[str], float]] = []

    for slot in slots:
        c = assignment.get(slot, None)
        if c is None:
            plan.append((slot.date, slot.produit, None, 0.0))
            continue

        cap = cap_by_slot[slot].get(c, 0.0)
        if cap < threshold:
            plan.append((slot.date, slot.produit, None, 0.0))
            continue

        avail = available_qty(batches.get(c, []), slot.date)
        v = min(cap, avail)

        if v >= threshold:
            used = consume(batches[c], slot.date, v)
            if used + 1e-12 >= threshold:
                total += used
                n_camp += 1
                plan.append((slot.date, slot.produit, c, used))
            else:
                plan.append((slot.date, slot.produit, None, 0.0))
        else:
            plan.append((slot.date, slot.produit, None, 0.0))

    obj = total - float(lambda_campaign) * n_camp
    return total, n_camp, obj, plan


# ==========================================================
# GREEDY (construction)
# ==========================================================

def greedy_construct(
    threshold: float,
    batches_by_chute: Dict[str, List[Batch]],
    cap_by_slot: Dict[Slot, Dict[str, float]],
    slots: List[Slot],
    alpha_urgency: float = 10.0,
    lambda_campaign: float = 1.0
) -> Dict[Slot, Optional[str]]:
    batches = deepcopy_batches(batches_by_chute)
    assignment: Dict[Slot, Optional[str]] = {}
    min_gain = max(float(threshold), float(lambda_campaign))

    for slot in slots:
        candidates = []
        for c, cap in cap_by_slot[slot].items():
            if cap < threshold:
                continue
            avail = available_qty(batches.get(c, []), slot.date)
            if avail < threshold:
                continue
            v = min(cap, avail)
            if v < min_gain:
                continue
            d = earliest_dlc_days(batches[c], slot.date)
            urgency = 0.0 if d is None else alpha_urgency / (1.0 + max(0, d))
            score = v + urgency
            candidates.append((score, c, v))

        if not candidates:
            assignment[slot] = None
            continue

        candidates.sort(reverse=True, key=lambda x: x[0])
        _, c_star, v_star = candidates[0]
        consume(batches[c_star], slot.date, v_star)
        assignment[slot] = c_star

    return assignment


# ==========================================================
# GRASP (construction RCL + tirage biaisé)
# ==========================================================

def grasp_construct(
    threshold: float,
    batches_by_chute: Dict[str, List[Batch]],
    cap_by_slot: Dict[Slot, Dict[str, float]],
    slots: List[Slot],
    rcl: int,
    alpha_urgency: float,
    lambda_campaign: float,
    rng: np.random.Generator,
    rcl_bias: float = 0.70
) -> Dict[Slot, Optional[str]]:
    batches = deepcopy_batches(batches_by_chute)
    assignment: Dict[Slot, Optional[str]] = {}
    min_gain = max(float(threshold), float(lambda_campaign))

    for slot in slots:
        candidates = []
        for c, cap in cap_by_slot[slot].items():
            if cap < threshold:
                continue
            avail = available_qty(batches.get(c, []), slot.date)
            if avail < threshold:
                continue

            v = min(cap, avail)
            if v < min_gain:
                continue

            d = earliest_dlc_days(batches[c], slot.date)
            urgency = 0.0 if d is None else alpha_urgency / (1.0 + max(0, d))
            score = v + urgency
            candidates.append((score, c, v))

        if not candidates:
            assignment[slot] = None
            continue

        candidates.sort(reverse=True, key=lambda x: x[0])
        top = candidates[: max(1, min(int(rcl), len(candidates)))]

        bias = float(rcl_bias)
        weights = np.array([(bias ** i) for i in range(len(top))], dtype=float)
        weights /= weights.sum()
        idx = int(rng.choice(len(top), p=weights))

        _, c_star, v_star = top[idx]
        consume(batches[c_star], slot.date, v_star)
        assignment[slot] = c_star

    return assignment


def grasp(
    threshold: float,
    batches_by_chute: Dict[str, List[Batch]],
    cap_by_slot: Dict[Slot, Dict[str, float]],
    slots: List[Slot],
    iterations: int = 200,
    rcl: int = 5,
    alpha_urgency: float = 10.0,
    lambda_campaign: float = 1.0,
    seed: int = 0,
    rcl_bias: float = 0.70
) -> Tuple[float, int, float, List[Tuple[pd.Timestamp, str, Optional[str], float]]]:
    rng = np.random.default_rng(seed)

    best_obj = -1e30
    best_total = 0.0
    best_camp = 0
    best_plan = None

    for _ in range(int(iterations)):
        ass = grasp_construct(
            threshold, batches_by_chute, cap_by_slot, slots,
            rcl=rcl,
            alpha_urgency=alpha_urgency,
            lambda_campaign=lambda_campaign,
            rng=rng,
            rcl_bias=rcl_bias
        )
        total, n_camp, obj, plan = evaluate_solution(
            ass, threshold, batches_by_chute, cap_by_slot, slots,
            lambda_campaign=lambda_campaign
        )
        if obj > best_obj + 1e-12:
            best_obj = obj
            best_total = total
            best_camp = n_camp
            best_plan = plan

    if best_plan is None:
        best_plan = [(s.date, s.produit, None, 0.0) for s in slots]
        best_total, best_camp, best_obj = 0.0, 0, 0.0

    return best_total, best_camp, best_obj, best_plan


# ==========================================================
# Outils Tabu
# ==========================================================

def _signature(slots: List[Slot], ass: Dict[Slot, Optional[str]]) -> Tuple[Optional[str], ...]:
    return tuple(ass.get(s, None) for s in slots)


def _tabu_key(tag: str, *items: Any) -> Tuple[Any, ...]:
    return (tag,) + tuple(items)


class LRUCache:
    def __init__(self, max_size: int = 5000):
        self.max_size = max(128, int(max_size))
        self.data: OrderedDict[Tuple[Optional[str], ...], Tuple[float, int, float]] = OrderedDict()

    def get(self, key):
        val = self.data.get(key)
        if val is not None:
            self.data.move_to_end(key)
        return val

    def put(self, key, value):
        self.data[key] = value
        self.data.move_to_end(key)
        if len(self.data) > self.max_size:
            self.data.popitem(last=False)


# ==========================================================
# TABU multi-opérateurs amélioré
# ==========================================================



def tabu_search_multiops_fast(
    start_assignment: Dict[Slot, Optional[str]],
    threshold: float,
    batches_by_chute: Dict[str, List[Batch]],
    cap_by_slot: Dict[Slot, Dict[str, float]],
    slots: List[Slot],
    lambda_campaign: float,
    rng: np.random.Generator,
    # vitesse / exploration
    tabu_iters: int = 80,
    tabu_tenure: int = 12,
    inactive_pool: int = 20,
    neighbor_samples: int = 24,
    candidate_chutes_topk: int = 4,
    swap_pairs: int = 12,
    reloc_pairs: int = 12,
    cache_size: int = 5000,
    patience: int = 20,
    # opérateurs
    enable_replace: bool = True,
    enable_swap: bool = True,
    enable_relocation: bool = False,
    enable_ruin_recreate: bool = False,  # legacy: ignoré volontairement
    ruin_k: int = 15,                    # legacy: ignoré volontairement
    ruin_freq: int = 10,                 # legacy: ignoré volontairement
    # améliorations état de l'art
    reactive_tabu: bool = True,
    tabu_tenure_min: int = 6,
    tabu_tenure_max: int = 28,
    revisit_window: int = 12,
    freq_penalty_weight: float = 0.02,
    intensify_after: int = 6,
    intensify_neighbors: int = 12,
    intensify_swap_pairs: int = 8,
    restart_after: int = 18,
) -> Dict[Slot, Optional[str]]:
    current = dict(start_assignment)
    _, _, best_obj, _ = evaluate_solution(
        current, threshold, batches_by_chute, cap_by_slot, slots, lambda_campaign=lambda_campaign
    )
    best_assignment = dict(current)

    # Pré-calculs
    top_candidates_by_slot: Dict[Slot, List[str]] = {}
    compatible_slots_by_chute: Dict[str, List[Slot]] = defaultdict(list)
    for s in slots:
        ordered = [
            c for c, cap in sorted(cap_by_slot[s].items(), key=lambda kv: kv[1], reverse=True)
            if cap >= threshold
        ]
        top_candidates_by_slot[s] = ordered[: max(1, int(candidate_chutes_topk))]
        for c in ordered:
            compatible_slots_by_chute[c].append(s)

    slot_rank = {s: i for i, s in enumerate(slots)}
    cache = LRUCache(cache_size)
    tabu: Dict[Tuple[Any, ...], int] = {}
    freq: Dict[Tuple[Any, ...], int] = defaultdict(int)
    seen_iter: Dict[Tuple[Optional[str], ...], int] = {}

    base_tenure = max(1, int(tabu_tenure))
    tenure = base_tenure
    last_improve_it = 0

    def eval_cached(ass: Dict[Slot, Optional[str]]) -> Tuple[float, int, float]:
        sig = _signature(slots, ass)
        val = cache.get(sig)
        if val is not None:
            return val
        total, camp, obj, _ = evaluate_solution(
            ass, threshold, batches_by_chute, cap_by_slot, slots, lambda_campaign=lambda_campaign
        )
        val = (total, camp, obj)
        cache.put(sig, val)
        return val

    def move_score(obj: float, move_key: Tuple[Any, ...]) -> float:
        if freq_penalty_weight <= 0:
            return obj
        return obj - float(freq_penalty_weight) * freq[move_key]

    def build_pool(ass: Dict[Slot, Optional[str]]) -> Tuple[List[Slot], List[Slot], List[Slot]]:
        active = [s for s in slots if ass.get(s, None) is not None]
        inactive = [s for s in slots if ass.get(s, None) is None]
        if inactive_pool > 0 and inactive:
            k = min(int(inactive_pool), len(inactive))
            inact_sample = rng.choice(inactive, size=k, replace=False).tolist()
        else:
            inact_sample = []
        return active + inact_sample, active, inactive

    def critical_slots(ass: Dict[Slot, Optional[str]], pool: List[Slot], k: int) -> List[Slot]:
        scored: List[Tuple[float, Slot]] = []
        for s in pool:
            cur = ass.get(s, None)
            cur_cap = 0.0 if cur is None else cap_by_slot[s].get(cur, 0.0)
            best_cap = cap_by_slot[s].get(top_candidates_by_slot[s][0], 0.0) if top_candidates_by_slot[s] else 0.0
            slack = max(0.0, best_cap - cur_cap)
            scored.append((slack, s))
        scored.sort(reverse=True, key=lambda x: x[0])
        return [s for _, s in scored[: max(1, min(k, len(scored)))]]

    for it in range(1, int(tabu_iters) + 1):
        if it - last_improve_it >= int(max(1, patience)):
            break

        # purge tabu expirés
        expired = [k for k, exp in tabu.items() if exp <= it]
        for k in expired:
            del tabu[k]

        pool, active_slots, inactive_slots = build_pool(current)
        if not pool:
            break

        # Intensification sur slots critiques après stagnation
        stagnation = it - last_improve_it
        if stagnation >= int(intensify_after):
            sampled_slots = critical_slots(current, pool, intensify_neighbors)
            local_swap_pairs = min(int(swap_pairs), int(intensify_swap_pairs))
            local_reloc_pairs = min(int(reloc_pairs), int(intensify_swap_pairs))
        else:
            sample_size = min(int(max(1, neighbor_samples)), len(pool))
            sampled_slots = rng.choice(pool, size=sample_size, replace=False).tolist()
            local_swap_pairs = int(swap_pairs)
            local_reloc_pairs = int(reloc_pairs)

        best_move_score = -1e30
        best_move_obj = -1e30
        best_move_key = None
        best_move_assignment = None

        # --------------------------------------------------
        # 1) REPLACE
        # --------------------------------------------------
        if enable_replace:
            for s in sampled_slots:
                options = [None] + top_candidates_by_slot.get(s, [])
                cur = current.get(s, None)
                for new_c in options:
                    if new_c == cur:
                        continue

                    move_key = _tabu_key("R", slot_rank[s], new_c)
                    trial = dict(current)
                    trial[s] = new_c
                    _, _, obj = eval_cached(trial)
                    if move_key in tabu and obj <= best_obj + 1e-12:
                        continue

                    sc = move_score(obj, move_key)
                    if sc > best_move_score + 1e-12:
                        best_move_score = sc
                        best_move_obj = obj
                        best_move_key = move_key
                        best_move_assignment = trial

        # --------------------------------------------------
        # 2) SWAP
        # --------------------------------------------------
        if enable_swap and len(sampled_slots) >= 2 and local_swap_pairs > 0:
            seen_pairs = set()
            n_slots = len(sampled_slots)
            max_pairs = min(local_swap_pairs, n_slots * (n_slots - 1) // 2)
            attempts = 0
            while len(seen_pairs) < max_pairs and attempts < 5 * max_pairs + 5:
                attempts += 1
                i1, i2 = sorted(rng.choice(n_slots, size=2, replace=False).tolist())
                if (i1, i2) in seen_pairs:
                    continue
                seen_pairs.add((i1, i2))
                s1, s2 = sampled_slots[i1], sampled_slots[i2]
                c1, c2 = current.get(s1, None), current.get(s2, None)
                if c1 == c2:
                    continue
                if c1 is not None and cap_by_slot[s2].get(c1, 0.0) < threshold:
                    continue
                if c2 is not None and cap_by_slot[s1].get(c2, 0.0) < threshold:
                    continue

                move_key = _tabu_key("S", min(slot_rank[s1], slot_rank[s2]), max(slot_rank[s1], slot_rank[s2]))
                trial = dict(current)
                trial[s1] = c2
                trial[s2] = c1
                _, _, obj = eval_cached(trial)
                if move_key in tabu and obj <= best_obj + 1e-12:
                    continue

                sc = move_score(obj, move_key)
                if sc > best_move_score + 1e-12:
                    best_move_score = sc
                    best_move_obj = obj
                    best_move_key = move_key
                    best_move_assignment = trial

        # --------------------------------------------------
        # 3) RELOCATION
        # --------------------------------------------------
        if enable_relocation and active_slots and inactive_slots and local_reloc_pairs > 0:
            act_pool = [s for s in sampled_slots if current.get(s, None) is not None]
            if not act_pool:
                act_pool = active_slots
            seen_moves = set()
            attempts = 0
            target_max = max(1, int(local_reloc_pairs))
            while len(seen_moves) < target_max and attempts < 8 * target_max + 8:
                attempts += 1
                s_from = rng.choice(act_pool)
                c = current.get(s_from, None)
                if c is None:
                    continue
                compat_targets = [s for s in compatible_slots_by_chute.get(c, []) if current.get(s, None) is None]
                if not compat_targets:
                    continue
                s_to = rng.choice(compat_targets)
                key_pair = (slot_rank[s_from], slot_rank[s_to], c)
                if key_pair in seen_moves:
                    continue
                seen_moves.add(key_pair)

                move_key = _tabu_key("M", slot_rank[s_from], slot_rank[s_to], c)
                trial = dict(current)
                trial[s_from] = None
                trial[s_to] = c
                _, _, obj = eval_cached(trial)
                if move_key in tabu and obj <= best_obj + 1e-12:
                    continue

                sc = move_score(obj, move_key)
                if sc > best_move_score + 1e-12:
                    best_move_score = sc
                    best_move_obj = obj
                    best_move_key = move_key
                    best_move_assignment = trial

        # --------------------------------------------------
        # aucun voisin admissible
        # --------------------------------------------------
        if best_move_assignment is None:
            break

        current = best_move_assignment
        freq[best_move_key] += 1
        tabu[best_move_key] = it + max(1, int(tenure))

        sig = _signature(slots, current)
        prev_seen = seen_iter.get(sig)
        if reactive_tabu and prev_seen is not None and (it - prev_seen) <= int(revisit_window):
            tenure = min(int(tabu_tenure_max), max(int(tabu_tenure_min), int(tenure) + 1))
        seen_iter[sig] = it

        _, _, cur_obj = eval_cached(current)
        if cur_obj > best_obj + 1e-12:
            best_obj = cur_obj
            best_assignment = dict(current)
            last_improve_it = it
            if reactive_tabu:
                tenure = max(int(tabu_tenure_min), int(tenure) - 1)

        # restart léger depuis la meilleure solution en cas de stagnation prolongée
        if restart_after > 0 and (it - last_improve_it) >= int(restart_after):
            current = dict(best_assignment)
            last_improve_it = it - int(intensify_after)
            if reactive_tabu:
                tenure = max(int(tabu_tenure_min), base_tenure)

    return best_assignment


# ==========================================================
# Wrapper GRASP + TABU
# ==========================================================

def grasp_tabu(
    threshold: float,
    batches_by_chute: Dict[str, List[Batch]],
    cap_by_slot: Dict[Slot, Dict[str, float]],
    slots: List[Slot],
    iterations: int = 8,
    rcl: int = 3,
    alpha_urgency: float = 10.0,
    lambda_campaign: float = 1.0,
    seed: int = 0,
    rcl_bias: float = 0.70,
    # tabu
    tabu_iters: int = 60,
    tabu_tenure: int = 12,
    start_from: str = "greedy",
    # vitesse / exploration
    inactive_pool: int = 20,
    neighbor_samples: int = 24,
    candidate_chutes_topk: int = 4,
    swap_pairs: int = 12,
    reloc_pairs: int = 12,
    cache_size: int = 5000,
    patience: int = 20,
    # opérateurs
    enable_replace: bool = True,
    enable_swap: bool = True,
    enable_relocation: bool = False,
    enable_ruin_recreate: bool = False,  # legacy: ignoré
    ruin_k: int = 15,                    # legacy: ignoré
    ruin_freq: int = 10,                 # legacy: ignoré
    # améliorations
    reactive_tabu: bool = True,
    tabu_tenure_min: int = 6,
    tabu_tenure_max: int = 28,
    revisit_window: int = 12,
    freq_penalty_weight: float = 0.02,
    restart_after: int = 18,
    intensify_after: int = 6,
    intensify_neighbors: int = 12,
    intensify_swap_pairs: int = 8,
) -> Tuple[float, int, float, List[Tuple[pd.Timestamp, str, Optional[str], float]]]:
    rng = np.random.default_rng(seed)

    best_obj = -1e30
    best_total = 0.0
    best_camp = 0
    best_plan = None

    for _ in range(int(iterations)):
        if start_from.lower().strip() == "greedy":
            start = greedy_construct(
                threshold, batches_by_chute, cap_by_slot, slots,
                alpha_urgency=alpha_urgency,
                lambda_campaign=lambda_campaign
            )
        else:
            start = grasp_construct(
                threshold, batches_by_chute, cap_by_slot, slots,
                rcl=rcl,
                alpha_urgency=alpha_urgency,
                lambda_campaign=lambda_campaign,
                rng=rng,
                rcl_bias=rcl_bias
            )

        improved = tabu_search_multiops_fast(
            start_assignment=start,
            threshold=threshold,
            batches_by_chute=batches_by_chute,
            cap_by_slot=cap_by_slot,
            slots=slots,
            lambda_campaign=lambda_campaign,
            rng=rng,
            tabu_iters=tabu_iters,
            tabu_tenure=tabu_tenure,
            inactive_pool=inactive_pool,
            neighbor_samples=neighbor_samples,
            candidate_chutes_topk=candidate_chutes_topk,
            swap_pairs=swap_pairs,
            reloc_pairs=reloc_pairs,
            cache_size=cache_size,
            patience=patience,
            enable_replace=enable_replace,
            enable_swap=enable_swap,
            enable_relocation=enable_relocation,
            enable_ruin_recreate=enable_ruin_recreate,
            ruin_k=ruin_k,
            ruin_freq=ruin_freq,
            reactive_tabu=reactive_tabu,
            tabu_tenure_min=tabu_tenure_min,
            tabu_tenure_max=tabu_tenure_max,
            revisit_window=revisit_window,
            freq_penalty_weight=freq_penalty_weight,
            restart_after=restart_after,
            intensify_after=intensify_after,
            intensify_neighbors=intensify_neighbors,
            intensify_swap_pairs=intensify_swap_pairs,
        )

        total, n_camp, obj, plan = evaluate_solution(
            improved, threshold, batches_by_chute, cap_by_slot, slots,
            lambda_campaign=lambda_campaign
        )

        if obj > best_obj + 1e-12:
            best_obj = obj
            best_total = total
            best_camp = n_camp
            best_plan = plan

    if best_plan is None:
        best_plan = [(s.date, s.produit, None, 0.0) for s in slots]
        best_total, best_camp, best_obj = 0.0, 0, 0.0

    return best_total, best_camp, best_obj, best_plan


# ==========================================================
# LNS autonome ajouté à la version TABU améliorée
# ==========================================================

def _top_chutes_by_slot(
    cap_by_slot: Dict[Slot, Dict[str, float]],
    slots: List[Slot],
    topk: int
) -> Dict[Slot, List[str]]:
    out: Dict[Slot, List[str]] = {}
    k = max(1, int(topk))
    for s in slots:
        caps = cap_by_slot[s]
        out[s] = sorted(caps.keys(), key=lambda c: caps[c], reverse=True)[:k]
    return out

def _compatible_slots_by_chute(
    cap_by_slot: Dict[Slot, Dict[str, float]],
    slots: List[Slot],
    threshold: float
) -> Dict[str, List[Slot]]:
    comp: Dict[str, List[Slot]] = {}
    thr = float(threshold)
    for s in slots:
        for c, cap in cap_by_slot[s].items():
            if cap + 1e-12 >= thr:
                comp.setdefault(c, []).append(s)
    return comp

def _apply_assignment_on_batches(
    assignment: Dict[Slot, Optional[str]],
    threshold: float,
    batches_by_chute: Dict[str, List[Batch]],
    cap_by_slot: Dict[Slot, Dict[str, float]],
    slots: List[Slot],
    skip_slots: Optional[set] = None
) -> Dict[str, List[Batch]]:
    batches = deepcopy_batches(batches_by_chute)
    skip_slots = skip_slots or set()
    thr = float(threshold)
    for s in slots:
        if s in skip_slots:
            continue
        c = assignment.get(s, None)
        if c is None:
            continue
        cap = cap_by_slot[s].get(c, 0.0)
        if cap + 1e-12 < thr:
            continue
        avail = available_qty(batches.get(c, []), s.date)
        v = min(cap, avail)
        if v + 1e-12 >= thr:
            consume(batches[c], s.date, v)
    return batches

def lns_repair_partial(
    base_assignment: Dict[Slot, Optional[str]],
    destroy_slots: List[Slot],
    threshold: float,
    batches_by_chute: Dict[str, List[Batch]],
    cap_by_slot: Dict[Slot, Dict[str, float]],
    slots: List[Slot],
    rng: np.random.Generator,
    lambda_campaign: float,
    alpha_urgency: float = 10.0,
    rcl: int = 4,
    rcl_bias: float = 0.80,
    top_chutes_by_slot: Optional[Dict[Slot, List[str]]] = None
) -> Dict[Slot, Optional[str]]:
    """
    LNS local repair: on fixe les slots non détruits, on reconstruit seulement
    les destroy_slots à partir du stock résiduel.
    """
    destroy_set = set(destroy_slots)
    trial = dict(base_assignment)
    for s in destroy_slots:
        trial[s] = None

    residual_batches = _apply_assignment_on_batches(
        trial, threshold, batches_by_chute, cap_by_slot, slots, skip_slots=destroy_set
    )

    min_gain = max(float(threshold), float(lambda_campaign))
    ordered_destroy = sorted(destroy_slots, key=lambda s: (s.date, s.produit))
    rrcl = max(1, int(rcl))
    bias = float(rcl_bias)

    for s in ordered_destroy:
        if top_chutes_by_slot is None:
            chute_pool = sorted(cap_by_slot[s].keys(), key=lambda c: cap_by_slot[s][c], reverse=True)
        else:
            chute_pool = top_chutes_by_slot[s]

        candidates = []
        for c in chute_pool:
            cap = cap_by_slot[s].get(c, 0.0)
            if cap + 1e-12 < threshold:
                continue
            avail = available_qty(residual_batches.get(c, []), s.date)
            if avail + 1e-12 < threshold:
                continue

            v = min(cap, avail)
            if v + 1e-12 < min_gain:
                continue

            d = earliest_dlc_days(residual_batches[c], s.date)
            urgency = 0.0 if d is None else alpha_urgency / (1.0 + max(0, d))
            score = v + urgency
            candidates.append((score, c, v))

        if not candidates:
            trial[s] = None
            continue

        candidates.sort(reverse=True, key=lambda x: x[0])
        top = candidates[: min(rrcl, len(candidates))]
        weights = np.array([(bias ** i) for i in range(len(top))], dtype=float)
        weights /= weights.sum()
        idx = int(rng.choice(len(top), p=weights))
        _, c_star, v_star = top[idx]

        consume(residual_batches[c_star], s.date, v_star)
        trial[s] = c_star

    return trial


# ==========================================================
# LNS heuristique autonome (sans Tabu)
# ==========================================================

def lns_search(
    start_assignment: Dict[Slot, Optional[str]],
    threshold: float,
    batches_by_chute: Dict[str, List[Batch]],
    cap_by_slot: Dict[Slot, Dict[str, float]],
    slots: List[Slot],
    lambda_campaign: float,
    rng: np.random.Generator,
    lns_iters: int = 80,
    lns_k: int = 15,
    lns_rcl: int = 4,
    lns_rcl_bias: float = 0.80,
    lns_alpha_urgency: float = 10.0,
    patience: int = 20,
    candidate_chutes_topk: int = 8,
    accept_non_improving: bool = False
) -> Dict[Slot, Optional[str]]:
    """
    Heuristique LNS autonome :
    - part d'une solution initiale
    - détruit k slots actifs
    - répare localement sur stock résiduel
    - conserve la meilleure solution trouvée
    """
    current = dict(start_assignment)
    _, _, current_obj, _ = evaluate_solution(
        current, threshold, batches_by_chute, cap_by_slot, slots, lambda_campaign=lambda_campaign
    )
    best_assignment = dict(current)
    best_obj = current_obj

    top_chutes_by_slot = _top_chutes_by_slot(cap_by_slot, slots, candidate_chutes_topk)
    no_improve = 0

    active_template = [s for s in slots if current.get(s, None) is not None]
    if not active_template:
        return best_assignment

    for _ in range(int(lns_iters)):
        active_slots = [s for s in slots if current.get(s, None) is not None]
        if not active_slots:
            break
        k = min(max(1, int(lns_k)), len(active_slots))
        destroy_slots = rng.choice(active_slots, size=k, replace=False).tolist()

        trial = lns_repair_partial(
            current, destroy_slots, threshold, batches_by_chute, cap_by_slot, slots,
            rng=rng,
            lambda_campaign=lambda_campaign,
            alpha_urgency=lns_alpha_urgency,
            rcl=lns_rcl,
            rcl_bias=lns_rcl_bias,
            top_chutes_by_slot=top_chutes_by_slot
        )
        _, _, trial_obj, _ = evaluate_solution(
            trial, threshold, batches_by_chute, cap_by_slot, slots, lambda_campaign=lambda_campaign
        )

        improved_best = trial_obj > best_obj + 1e-12
        improved_current = trial_obj > current_obj + 1e-12

        if improved_best:
            best_obj = trial_obj
            best_assignment = dict(trial)

        if improved_current or accept_non_improving:
            current = trial
            current_obj = trial_obj

        if improved_best:
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= int(patience):
                break

    return best_assignment


def grasp_lns(
    threshold: float,
    batches_by_chute: Dict[str, List[Batch]],
    cap_by_slot: Dict[Slot, Dict[str, float]],
    slots: List[Slot],
    iterations: int = 10,
    rcl: int = 3,
    alpha_urgency: float = 10.0,
    lambda_campaign: float = 1.0,
    seed: int = 0,
    rcl_bias: float = 0.70,
    start_from: str = "greedy",
    lns_iters: int = 80,
    lns_k: int = 15,
    lns_rcl: int = 4,
    lns_rcl_bias: float = 0.80,
    lns_alpha_urgency: float = 10.0,
    patience: int = 20,
    candidate_chutes_topk: int = 8,
    accept_non_improving: bool = False
) -> Tuple[float, int, float, List[Tuple[pd.Timestamp, str, Optional[str], float]]]:
    rng = np.random.default_rng(seed)

    best_obj = -1e30
    best_total = 0.0
    best_camp = 0
    best_plan = None

    for _ in range(int(iterations)):
        if start_from.lower().strip() == "greedy":
            start = greedy_construct(
                threshold, batches_by_chute, cap_by_slot, slots,
                alpha_urgency=alpha_urgency,
                lambda_campaign=lambda_campaign
            )
        else:
            start = grasp_construct(
                threshold, batches_by_chute, cap_by_slot, slots,
                rcl=rcl,
                alpha_urgency=alpha_urgency,
                lambda_campaign=lambda_campaign,
                rng=rng,
                rcl_bias=rcl_bias
            )

        improved = lns_search(
            start, threshold, batches_by_chute, cap_by_slot, slots,
            lambda_campaign=lambda_campaign,
            rng=rng,
            lns_iters=lns_iters,
            lns_k=lns_k,
            lns_rcl=lns_rcl,
            lns_rcl_bias=lns_rcl_bias,
            lns_alpha_urgency=lns_alpha_urgency,
            patience=patience,
            candidate_chutes_topk=candidate_chutes_topk,
            accept_non_improving=accept_non_improving
        )

        total, n_camp, obj, plan = evaluate_solution(
            improved, threshold, batches_by_chute, cap_by_slot, slots,
            lambda_campaign=lambda_campaign
        )
        if obj > best_obj + 1e-12:
            best_obj = obj
            best_total = total
            best_camp = n_camp
            best_plan = plan

    if best_plan is None:
        best_plan = [(s.date, s.produit, None, 0.0) for s in slots]
        best_total, best_camp, best_obj = 0.0, 0, 0.0

    return best_total, best_camp, best_obj, best_plan


# ==========================================================
# GRASP_TABU configurable (démarre depuis greedy ou grasp)
# ==========================================================

