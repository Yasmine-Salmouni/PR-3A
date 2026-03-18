# -*- coding: utf-8 -*-
"""
reincorp_heuristic.py

Fonctions:
- load_data
- evaluate_solution
- greedy_construct
- grasp (RCL + tirage biaisé)
- grasp_tabu (Tabu multi-opérateurs configurable)

Objectif:
    obj = volume_total - lambda_campaign * nb_campagnes

Notes:
- Construction greedy/grasp filtre les campagnes "non rentables":
    v >= max(seuil, lambda_campaign)

Tabu multi-opérateurs:
- replace: changer chute d'un slot
- swap: échanger deux slots
- relocation: déplacer une chute d'un slot actif vers un slot vide
- ruin_recreate (mini-LNS): détruire k slots et réparer
"""

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
# TABU multi-opérateurs (FAST)
# ==========================================================

def _signature(slots: List[Slot], ass: Dict[Slot, Optional[str]]) -> Tuple[Optional[str], ...]:
    return tuple(ass.get(s, None) for s in slots)

def _tabu_key(tag: str, *items: Any) -> Tuple[Any, ...]:
    return (tag,) + tuple(items)

def tabu_search_multiops_fast(
    start_assignment: Dict[Slot, Optional[str]],
    threshold: float,
    batches_by_chute: Dict[str, List[Batch]],
    cap_by_slot: Dict[Slot, Dict[str, float]],
    slots: List[Slot],
    lambda_campaign: float,
    rng: np.random.Generator,
    # contrôles vitesse
    tabu_iters: int = 80,
    tabu_tenure: int = 15,
    inactive_pool: int = 25,
    neighbor_samples: int = 35,
    candidate_chutes_topk: int = 8,
    swap_pairs: int = 20,
    reloc_pairs: int = 25,
    cache_size: int = 2000,
    patience: int = 20,
    # opérateurs
    enable_replace: bool = True,
    enable_swap: bool = True,
    enable_relocation: bool = False,
    enable_ruin_recreate: bool = False,
    ruin_k: int = 15,
    ruin_freq: int = 10
) -> Dict[Slot, Optional[str]]:
    """
    Tabu Search configurable:
    - replace / swap / relocation / ruin&recreate
    - aspiration + best-keeping
    """
    current = dict(start_assignment)

    # best global
    _, _, best_obj, _ = evaluate_solution(
        current, threshold, batches_by_chute, cap_by_slot, slots, lambda_campaign=lambda_campaign
    )
    best_assignment = dict(current)

    tabu: Dict[Tuple[Any, ...], int] = {}
    eval_cache: Dict[Tuple[Optional[str], ...], Tuple[float, int, float]] = {}

    def eval_cached(ass: Dict[Slot, Optional[str]]) -> Tuple[float, int, float]:
        sig = _signature(slots, ass)
        if sig in eval_cache:
            return eval_cache[sig]
        total, camp, obj, _ = evaluate_solution(
            ass, threshold, batches_by_chute, cap_by_slot, slots, lambda_campaign=lambda_campaign
        )
        if len(eval_cache) >= cache_size:
            eval_cache.clear()
        eval_cache[sig] = (total, camp, obj)
        return total, camp, obj

    def build_pool(ass: Dict[Slot, Optional[str]]) -> Tuple[List[Slot], List[Slot], List[Slot]]:
        active = [s for s in slots if ass.get(s, None) is not None]
        inactive = [s for s in slots if ass.get(s, None) is None]
        if inactive_pool > 0 and inactive:
            inactive_sample = rng.choice(inactive, size=min(inactive_pool, len(inactive)), replace=False).tolist()
        else:
            inactive_sample = []
        pool = active + inactive_sample
        return pool, active, inactive

    last_improve_it = 0

    for it in range(1, int(tabu_iters) + 1):
        if it - last_improve_it >= int(patience):
            break

        # purge tabu expirés
        expired = [k for k, exp in tabu.items() if exp <= it]
        for k in expired:
            del tabu[k]

        pool, active_slots, inactive_slots = build_pool(current)
        if not pool:
            break

        sample_size = min(int(neighbor_samples), len(pool))
        sampled_slots = rng.choice(pool, size=sample_size, replace=False).tolist()

        best_move_obj = -1e30
        best_move_key = None
        best_move_assignment = None

        # ------------------------------
        # REPLACE
        # ------------------------------
        if enable_replace:
            for s in sampled_slots:
                caps = cap_by_slot[s]
                chutes = sorted(caps.keys(), key=lambda c: caps[c], reverse=True)[:int(candidate_chutes_topk)]
                options = [None] + chutes

                for new_c in options:
                    if new_c == current.get(s, None):
                        continue

                    move_key = _tabu_key("R", s.date, s.produit, new_c)
                    is_tabu = move_key in tabu

                    trial = dict(current)
                    trial[s] = new_c

                    _, _, obj = eval_cached(trial)

                    # aspiration
                    if is_tabu and obj <= best_obj + 1e-12:
                        continue

                    if obj > best_move_obj + 1e-12:
                        best_move_obj = obj
                        best_move_key = move_key
                        best_move_assignment = trial

        # ------------------------------
        # SWAP (filtré compatibilité)
        # ------------------------------
        if enable_swap and len(sampled_slots) >= 2:
            for _ in range(int(swap_pairs)):
                s1, s2 = rng.choice(sampled_slots, size=2, replace=False).tolist()
                c1 = current.get(s1, None)
                c2 = current.get(s2, None)
                if c1 == c2:
                    continue

                # filtre compat: éviter swaps inutiles (cap=0)
                if c1 is not None and cap_by_slot[s2].get(c1, 0.0) < threshold:
                    continue
                if c2 is not None and cap_by_slot[s1].get(c2, 0.0) < threshold:
                    continue

                move_key = _tabu_key("S", s1.date, s1.produit, s2.date, s2.produit)
                is_tabu = move_key in tabu

                trial = dict(current)
                trial[s1] = c2
                trial[s2] = c1

                _, _, obj = eval_cached(trial)

                if is_tabu and obj <= best_obj + 1e-12:
                    continue

                if obj > best_move_obj + 1e-12:
                    best_move_obj = obj
                    best_move_key = move_key
                    best_move_assignment = trial

        # ------------------------------
        # RELOCATION (actif -> vide)
        # ------------------------------
        if enable_relocation and active_slots and inactive_slots:
            # on pioche dans un petit sous-ensemble pour rester rapide
            act_sample = rng.choice(active_slots, size=min(len(active_slots), sample_size), replace=False).tolist()
            inact_sample = rng.choice(inactive_slots, size=min(len(inactive_slots), sample_size), replace=False).tolist()

            for _ in range(int(reloc_pairs)):
                s_from = rng.choice(act_sample)
                c = current.get(s_from, None)
                if c is None:
                    continue
                s_to = rng.choice(inact_sample)

                # filtre compat: chute c doit être possible sur s_to
                if cap_by_slot[s_to].get(c, 0.0) < threshold:
                    continue

                move_key = _tabu_key("M", s_from.date, s_from.produit, s_to.date, s_to.produit, c)
                is_tabu = move_key in tabu

                trial = dict(current)
                trial[s_from] = None
                trial[s_to] = c

                _, _, obj = eval_cached(trial)

                if is_tabu and obj <= best_obj + 1e-12:
                    continue

                if obj > best_move_obj + 1e-12:
                    best_move_obj = obj
                    best_move_key = move_key
                    best_move_assignment = trial

        # ------------------------------
        # RUIN & RECREATE (mini-LNS)
        # 1 candidat toutes les ruin_freq itérations
        # ------------------------------
        if enable_ruin_recreate and active_slots and (it % int(ruin_freq) == 0):
            k = min(int(ruin_k), len(active_slots))
            destroy_slots = rng.choice(active_slots, size=k, replace=False).tolist()

            trial = dict(current)
            for s in destroy_slots:
                trial[s] = None

            # repair: on reconstruit une affectation complète puis on prend seulement les slots détruits
            repair = grasp_construct(
                threshold, batches_by_chute, cap_by_slot, slots,
                rcl=5,
                alpha_urgency=10.0,
                lambda_campaign=lambda_campaign,
                rng=rng,
                rcl_bias=0.80
            )
            for s in destroy_slots:
                trial[s] = repair.get(s, None)

            move_key = _tabu_key("LNS", it)
            is_tabu = move_key in tabu

            _, _, obj = eval_cached(trial)
            if (not is_tabu) or (obj > best_obj + 1e-12):
                if obj > best_move_obj + 1e-12:
                    best_move_obj = obj
                    best_move_key = move_key
                    best_move_assignment = trial

        # appliquer
        if best_move_assignment is None:
            break

        current = best_move_assignment
        tabu[best_move_key] = it + int(tabu_tenure)

        # update best
        _, _, obj = eval_cached(current)
        if obj > best_obj + 1e-12:
            best_obj = obj
            best_assignment = dict(current)
            last_improve_it = it

    return best_assignment


# ==========================================================
# GRASP_TABU configurable (démarre depuis greedy ou grasp)
# ==========================================================

def grasp_tabu(
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
    tabu_iters: int = 80,
    tabu_tenure: int = 15,
    start_from: str = "greedy",
    # opérateurs
    enable_replace: bool = True,
    enable_swap: bool = True,
    enable_relocation: bool = False,
    enable_ruin_recreate: bool = False,
    ruin_k: int = 15,
    ruin_freq: int = 10
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
            start,
            threshold, batches_by_chute, cap_by_slot, slots,
            lambda_campaign=lambda_campaign,
            rng=rng,
            tabu_iters=tabu_iters,
            tabu_tenure=tabu_tenure,
            enable_replace=enable_replace,
            enable_swap=enable_swap,
            enable_relocation=enable_relocation,
            enable_ruin_recreate=enable_ruin_recreate,
            ruin_k=ruin_k,
            ruin_freq=ruin_freq
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