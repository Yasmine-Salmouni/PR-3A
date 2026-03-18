import csv
import itertools
import os
import re
import signal
import sys
import time
from collections import defaultdict, OrderedDict
from datetime import datetime
import numpy as np


# =============================================================================
# Chargement des données (identique au modèle initial)
# =============================================================================

def extract_product_number(product_name):
    """
    Extrait le numéro numérique d'un nom de produit (ex: "produit 1" -> 1)
    """
    match = re.search(r'\d+', product_name)
    return int(match.group()) if match else 0

def load_stock_proj_chute(filepath):
    """
    Charge stock_proj_chute(c, dc) depuis le CSV.

    Retourne:
        stock : dict[(c, dc)] -> float   où dc = (dmc, dlc)
        chute_windows : dict[c] -> list[(dmc, dlc)]
    """
    stock = {}
    chute_windows = defaultdict(list)

    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            c = row["C Chute"].strip()
            dmc = datetime.strptime(row["DC DMC"].strip(), "%Y-%m-%d %H:%M:%S")
            dlc = datetime.strptime(row["DC DLC"].strip(), "%Y-%m-%d %H:%M:%S")
            stock_kg = float(row["Stock_proj_chute(c,dc)"].strip())

            dc = (dmc, dlc)
            stock[(c, dc)] = stock_kg
            if dc not in chute_windows[c]:
                chute_windows[c].append(dc)

    return stock, dict(chute_windows)


def load_seuil_reincorpo_mini(filepath):
    """
    Charge le seuil minimal de ré-incorporation depuis le CSV.

    Retourne:
        seuil_mini : float (en kg)
    """
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=";")
        next(reader)  # sauter l'en-tête
        row = next(reader)
        seuil_mini = float(row[1].strip())
    return seuil_mini


def load_plan_production(filepath):
    """
    Charge reincorpo_maxi(p, dp, c) depuis le CSV.

    Retourne:
        reincorpo_maxi : dict[(p, dp, c)] -> float
    """
    reincorpo_maxi = {}

    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            dp = datetime.strptime(row["DP Date"].strip(), "%Y-%m-%d %H:%M:%S")
            p = row["P Produit"].strip()
            c = row["c Chute"].strip()
            maxi = float(row["reincorpo_maxi(p, dp, c)"].strip())

            reincorpo_maxi[(p, dp, c)] = maxi

    return reincorpo_maxi


# =============================================================================
# Heuristique LNS (Large Neighborhood Search)
# =============================================================================

def greedy_construct(
    threshold: float,
    stock_proj_chute: dict,
    reincorpo_maxi: dict,
    options: dict,
    slots: list,
    alpha_urgency: float = 10.0,
    lambda_campaign: float = 1.0
) -> dict:
    """Construction gloutonne pour solution initiale."""
    remaining_stock = dict(stock_proj_chute)
    assignment = {}
    used_activations = defaultdict(set)
    
    min_gain = max(float(threshold), float(lambda_campaign))
    
    for (p, dp) in slots:
        if len(used_activations[(p, dp)]) >= 1:
            continue
            
        candidates = []
        for (c, dc, maxi) in options.get((p, dp), []):
            if maxi < threshold:
                continue
            avail = remaining_stock.get((c, dc), 0.0)
            if avail < threshold:
                continue
            v = min(maxi, avail)
            if v < min_gain:
                continue
            
            dmc, dlc = dc
            days_to_expiry = (dlc - dp).days
            urgency = 0.0 if days_to_expiry < 0 else alpha_urgency / (1.0 + max(0, days_to_expiry))
            score = v + urgency
            candidates.append((score, c, dc, v))
        
        if not candidates:
            continue
            
        candidates.sort(reverse=True, key=lambda x: x[0])
        _, best_c, best_dc, best_v = candidates[0]
        
        assignment[(p, dp, best_c)] = best_v
        remaining_stock[(best_c, best_dc)] -= best_v
        used_activations[(p, dp)].add(best_c)
    
    return assignment


def evaluate_solution(
    assignment: dict,
    stock_proj_chute: dict,
    reincorpo_maxi: dict,
    lambda_campaign: float = 1.0
) -> tuple:
    """Évalue une solution et retourne (total, n_activations, objectif)."""
    total = sum(assignment.values())
    n_activations = len(assignment)
    obj = total - lambda_campaign * n_activations
    return total, n_activations, obj


def grasp_construct(
    threshold: float,
    stock_proj_chute: dict,
    reincorpo_maxi: dict,
    options: dict,
    slots: list,
    rcl: int,
    alpha_urgency: float,
    lambda_campaign: float,
    rng: np.random.Generator,
    rcl_bias: float = 0.70
) -> dict:
    """Construction GRASP pour LNS."""
    remaining_stock = dict(stock_proj_chute)
    assignment = {}
    used_activations = defaultdict(set)
    
    min_gain = max(float(threshold), float(lambda_campaign))

    for (p, dp) in slots:
        if len(used_activations[(p, dp)]) >= 1:
            continue
            
        candidates = []
        for (c, dc, maxi) in options.get((p, dp), []):
            if maxi < threshold:
                continue
            avail = remaining_stock.get((c, dc), 0.0)
            if avail < threshold:
                continue
            v = min(maxi, avail)
            if v < min_gain:
                continue
            
            dmc, dlc = dc
            days_to_expiry = (dlc - dp).days
            urgency = 0.0 if days_to_expiry < 0 else alpha_urgency / (1.0 + max(0, days_to_expiry))
            score = v + urgency
            candidates.append((score, c, dc, v))

        if not candidates:
            continue

        candidates.sort(reverse=True, key=lambda x: x[0])
        top_candidates = candidates[: max(1, min(int(rcl), len(candidates)))]

        bias = float(rcl_bias)
        weights = np.array([(bias ** i) for i in range(len(top_candidates))], dtype=float)
        weights /= weights.sum()
        idx = int(rng.choice(len(top_candidates), p=weights))

        _, chosen_c, chosen_dc, chosen_v = top_candidates[idx]

        assignment[(p, dp, chosen_c)] = chosen_v
        remaining_stock[(chosen_c, chosen_dc)] -= chosen_v
        used_activations[(p, dp)].add(chosen_c)

    return assignment


def _apply_assignment_on_batches(
    assignment: dict,
    stock_proj_chute: dict,
    reincorpo_maxi: dict,
    options: dict,
    skip_slots: set = None
) -> dict:
    """Applique une assignation sur les stocks et retourne les stocks résiduels."""
    residual_stock = dict(stock_proj_chute)
    skip_slots = skip_slots or set()
    
    for (p, dp, c), vol in assignment.items():
        if (p, dp) in skip_slots:
            continue
        # Trouver la fenêtre (dc) correspondante pour cette chute
        for (c2, dc, maxi) in options.get((p, dp), []):
            if c2 == c:
                residual_stock[(c, dc)] = max(0.0, residual_stock.get((c, dc), 0.0) - vol)
                break
    
    return residual_stock


def lns_repair_partial(
    base_assignment: dict,
    destroy_slots: list,
    stock_proj_chute: dict,
    reincorpo_maxi: dict,
    options: dict,
    slots: list,
    threshold: float,
    rng: np.random.Generator,
    lambda_campaign: float,
    alpha_urgency: float = 10.0,
    rcl: int = 4,
    rcl_bias: float = 0.80
) -> dict:
    """
    LNS local repair: on fixe les slots non détruits, on reconstruit seulement
    les destroy_slots à partir du stock résiduel.
    """
    destroy_set = set((p, dp) for p, dp in destroy_slots)
    trial = dict(base_assignment)
    
    # Supprimer les assignations des slots détruits
    for (p, dp, c) in list(trial.keys()):
        if (p, dp) in destroy_set:
            del trial[(p, dp, c)]

    # Calculer les stocks résiduels après avoir appliqué l'assignation de base
    residual_stock = _apply_assignment_on_batches(trial, stock_proj_chute, reincorpo_maxi, options, skip_slots=destroy_set)

    min_gain = max(float(threshold), float(lambda_campaign))
    ordered_destroy = sorted(destroy_slots, key=lambda s: (extract_product_number(s[0]), s[1]))
    rrcl = max(1, int(rcl))
    bias = float(rcl_bias)

    for (p, dp) in ordered_destroy:
        candidates = []
        for (c, dc, maxi) in options.get((p, dp), []):
            if maxi < threshold:
                continue
            avail = residual_stock.get((c, dc), 0.0)
            if avail < threshold:
                continue
            v = min(maxi, avail)
            if v < min_gain:
                continue
            
            dmc, dlc = dc
            days_to_expiry = (dlc - dp).days
            urgency = 0.0 if days_to_expiry < 0 else alpha_urgency / (1.0 + max(0, days_to_expiry))
            score = v + urgency
            candidates.append((score, c, dc, v))

        if not candidates:
            continue

        candidates.sort(reverse=True, key=lambda x: x[0])
        top = candidates[: min(rrcl, len(candidates))]
        weights = np.array([(bias ** i) for i in range(len(top))], dtype=float)
        weights /= weights.sum()
        idx = int(rng.choice(len(top), p=weights))

        _, chosen_c, chosen_dc, chosen_v = top[idx]

        trial[(p, dp, chosen_c)] = chosen_v
        residual_stock[(chosen_c, chosen_dc)] = max(0.0, residual_stock.get((chosen_c, chosen_dc), 0.0) - chosen_v)

    return trial


def lns_search(
    start_assignment: dict,
    stock_proj_chute: dict,
    reincorpo_maxi: dict,
    options: dict,
    slots: list,
    threshold: float,
    lambda_campaign: float,
    rng: np.random.Generator,
    lns_iters: int = 80,
    lns_k: int = 15,
    lns_rcl: int = 4,
    lns_rcl_bias: float = 0.80,
    lns_alpha_urgency: float = 10.0,
    patience: int = 20,
    accept_non_improving: bool = False
) -> dict:
    """
    Heuristique LNS autonome :
    - part d'une solution initiale
    - détruit k slots actifs
    - répare localement sur stock résiduel
    - conserve la meilleure solution trouvée
    """
    current = dict(start_assignment)
    _, _, current_obj = evaluate_solution(current, stock_proj_chute, reincorpo_maxi, lambda_campaign)
    best_assignment = dict(current)
    best_obj = current_obj

    no_improve = 0

    active_template = [(p, dp) for (p, dp, c) in current.keys()]
    if not active_template:
        return best_assignment

    for _ in range(int(lns_iters)):
        active_slots = [(p, dp) for (p, dp, c) in current.keys()]
        if not active_slots:
            break
        
        k = min(max(1, int(lns_k)), len(active_slots))
        destroy_slots = rng.choice(active_slots, size=k, replace=False).tolist()

        trial = lns_repair_partial(
            current, destroy_slots, stock_proj_chute, reincorpo_maxi, options, slots,
            threshold, rng, lambda_campaign, alpha_urgency=lns_alpha_urgency,
            rcl=lns_rcl, rcl_bias=lns_rcl_bias
        )
        
        _, _, trial_obj = evaluate_solution(trial, stock_proj_chute, reincorpo_maxi, lambda_campaign)

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


# =============================================================================
# Relaxation Lagrangienne avec LNS
# =============================================================================

def solve_lagrangian_lns(stock_file, plan_file, seuil_reincorpo_mini=100.0,
                        alpha_init=0.1, epsilon=0.01, max_iterations=100,
                        # Paramètres LNS
                        lns_iters=80, lns_k=15, lns_rcl=4, lns_rcl_bias=0.80,
                        lns_alpha_urgency=10.0, patience=20,
                        accept_non_improving=False, start_from="greedy",
                        # Paramètres communs
                        alpha_urgency=10.0, lambda_campaign=1.0, seed=0):
    """
    Résout le modèle de ré-incorporation par relaxation lagrangienne
    avec heuristique LNS pour la réparation.
    """
    
    # Variable globale pour gérer l'interruption
    interrupted = False
    
    def signal_handler(sig, frame):
        nonlocal interrupted
        interrupted = True
        print(f"\n\n  Interruption par l'utilisateur (Ctrl+C) à l'itération {iteration}")
        print("  Affichage des résultats partiels...")
    
    signal.signal(signal.SIGINT, signal_handler)

    # -----------------------------------------------------------------
    # 1. Chargement des données
    # -----------------------------------------------------------------
    stock_proj_chute, chute_windows = load_stock_proj_chute(stock_file)
    reincorpo_maxi = load_plan_production(plan_file)

    # -----------------------------------------------------------------
    # Pré-calcul : options valides par (p, dp)
    # -----------------------------------------------------------------
    options = defaultdict(list)
    for (p, dp, c), maxi in reincorpo_maxi.items():
        if c not in chute_windows:
            continue
        if maxi < seuil_reincorpo_mini:
            continue
        for dc in chute_windows[c]:
            dmc, dlc = dc
            if dmc <= dp <= dlc:
                options[(p, dp)].append((c, dc, maxi))

    # Trier les slots par ordre numérique sur produits puis chronologique
    slots = sorted(options.keys(), key=lambda x: (extract_product_number(x[0]), x[1]))

    print("=" * 70)
    print("RELAXATION LAGRANGIENNE — HEURISTIQUE LNS")
    print("=" * 70)
    print(f"  Nombre de slots (p, dp)           : {len(slots)}")
    print(f"  Seuil mini (kg)                   : {seuil_reincorpo_mini}")
    print(f"  LNS iterations                   : {lns_iters}")
    print(f"  LNS k (destruction)              : {lns_k}")
    print(f"  LNS RCL size                     : {lns_rcl}")
    print(f"  LNS RCL bias                     : {lns_rcl_bias}")
    print(f"  LNS alpha urgency                : {lns_alpha_urgency}")
    print(f"  Start from                       : {start_from}")
    print(f"  Alpha urgency                    : {alpha_urgency}")
    print(f"  Lambda campaign                  : {lambda_campaign}")
    print(f"  Seed                             : {seed}")
    print(f"  Epsilon (critère d'arrêt)        : {epsilon}")
    print(f"  Nombre maximum d'itérations       : {max_iterations}")
    print()
    print("Résolution en cours...")
    print()

    # -----------------------------------------------------------------
    # Step 1 & 2 : Solution initiale valide
    # -----------------------------------------------------------------
    if start_from.lower().strip() == "greedy":
        start_assignment = greedy_construct(
            seuil_reincorpo_mini, stock_proj_chute, reincorpo_maxi, 
            options, slots, alpha_urgency, lambda_campaign
        )
    else:
        rng_start = np.random.default_rng(seed)
        start_assignment = grasp_construct(
            seuil_reincorpo_mini, stock_proj_chute, reincorpo_maxi, options, slots,
            rcl=3, alpha_urgency=alpha_urgency, lambda_campaign=lambda_campaign,
            rng=rng_start, rcl_bias=0.70
        )
    
    Z_best_primal = sum(start_assignment.values()) - len(start_assignment) * lambda_campaign
    best_assignment = dict(start_assignment)

    print(f"  Solution initiale {start_from.upper()} Z_best_primal : {Z_best_primal:.2f}")

    # -----------------------------------------------------------------
    # Step 3 : Initialisation des multiplicateurs et paramètres
    # -----------------------------------------------------------------
    k = 1
    alpha = alpha_init
    lam = {}
    for (c, dc) in stock_proj_chute:
        lam[(c, dc)] = 0.1

    no_improve_count = 0
    best_Z_dual_ever = float("inf")
    t_start = time.time()
    rng = np.random.default_rng(seed)

    for iteration in range(1, max_iterations + 1):
        
        if interrupted:
            break

        # =============================================================
        # Step 4 & 5 : Résolution des sous-problèmes
        # =============================================================
        R_brut = {}
        O_brut = {}
        usage_brut = defaultdict(float)
        sum_best_scores = 0.0

        for (p, dp), opts in sorted(options.items(), 
                                  key=lambda x: (extract_product_number(x[0][0]), x[0][1])):
            best_score = 0.0
            best_choice = None
            best_volume = 0.0

            for (c, dc, maxi) in opts:
                stock_disponible = stock_proj_chute.get((c, dc), 0.0)
                volume_possible = min(stock_disponible, maxi)
                
                score = (1.0 - lam[(c, dc)]) * volume_possible - lambda_campaign
                if score > best_score:
                    best_score = score
                    best_choice = (c, dc, maxi)
                    best_volume = volume_possible

            if best_choice is not None and best_volume > 0:
                c, dc, maxi = best_choice
                O_brut[(p, dp, c)] = 1
                R_brut[(c, dc, p, dp)] = best_volume
                usage_brut[(c, dc)] += best_volume
                sum_best_scores += best_score

        # =============================================================
        # Step 6 : Borne Duale
        # =============================================================
        lambda_stock_sum = sum(lam[key] * stock_proj_chute[key] for key in stock_proj_chute)
        Z_dual = lambda_stock_sum + sum_best_scores

        # =============================================================
        # Step 7 : Heuristique LNS pour réparation (appliquée à chaque itération)
        # =============================================================
        lns_repair = lns_search(
            best_assignment, stock_proj_chute, reincorpo_maxi, options, slots,
            seuil_reincorpo_mini, lambda_campaign, rng,
            lns_iters=lns_iters, lns_k=lns_k, lns_rcl=lns_rcl, lns_rcl_bias=lns_rcl_bias,
            lns_alpha_urgency=lns_alpha_urgency, patience=patience,
            accept_non_improving=accept_non_improving
        )
        
        lns_obj = sum(lns_repair.values()) - len(lns_repair) * lambda_campaign
        if lns_obj > Z_best_primal:
            Z_best_primal = lns_obj
            best_assignment = dict(lns_repair)

        # =============================================================
        # Step 8 : Sous-gradient et Mise à jour
        # =============================================================
        gap = Z_dual - Z_best_primal
        if gap <= epsilon:
            print(f"\n  Convergence atteinte (gap = {gap:.4f} ≤ ε = {epsilon})")
            break

        if Z_dual < best_Z_dual_ever - 1e-6:
            best_Z_dual_ever = Z_dual
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= 5:
                alpha = alpha / 2.0
                no_improve_count = 0
                print(f"  alpha réduit à {alpha:.4f} après 5 itérations sans battre le record")

        # Calcul des erreurs g
        g = {}
        sum_g2 = 0.0
        for (c, dc) in stock_proj_chute:
            g[(c, dc)] = usage_brut.get((c, dc), 0.0) - stock_proj_chute[(c, dc)]
            sum_g2 += g[(c, dc)] ** 2

        if sum_g2 < 1e-12:
            print(f"\n  Sous-gradient nul à l'itération {iteration}.")
            break

        step = (alpha * (Z_dual - Z_best_primal)) / sum_g2

        for (c, dc) in lam:
            lam[(c, dc)] = max(0.0, lam[(c, dc)] + step * g[(c, dc)])

        if interrupted:
            break

    t_end = time.time()

    # Statut et temps
    if interrupted:
        status = "INTERRUPTED"
        print(f"\n  Statut : {status} (arrêt par l'utilisateur)")
    elif gap <= epsilon:
        status = "OPTIMAL"
    elif iteration >= max_iterations:
        status = "LIMIT_REACHED"
        print(f"\n  Statut : {status} (limite d'itérations atteinte)")
    else:
        status = "NOT_SOLVED"
    
    if not interrupted:
        print(f"Statut : {status}")
    print(f"Temps de résolution : {t_end - t_start:.3f} secondes")
    print(f"Nombre d'itérations : {iteration}")

    # -----------------------------------------------------------------
    # Step 9 : Résultats finaux
    # -----------------------------------------------------------------
    total_reincorpore = sum(best_assignment.values())
    total_stock = sum(stock_proj_chute.values())
    nb_activations = len(best_assignment)
    
    print()
    print("=" * 70)
    if interrupted:
        print("RÉSULTATS PARTIELS (meilleure solution trouvée avant interruption)")
    else:
        print("RÉSULTATS FINAUX (meilleure solution primale)")
    print("=" * 70)
    print(f"  Valeur objectif (Z_dual)    : {Z_dual:.2f}")
    print(f"  Meilleure valeur primale   : {Z_best_primal:.2f}")
    print(f"  Gap final                   : {gap:.4f}")
    print(f"  Volume total ré-incorporé   : {total_reincorpore:.2f} kg")
    print(f"  Stock total initial disponible : {total_stock:.2f} kg")
    print(f"  Taux de ré-incorporation    : {100 * total_reincorpore / total_stock:.1f} %")
    print(f"  Nombre d'activations        : {nb_activations}")
    print(f"  Heuristique utilisée        : LNS (iters={lns_iters}, k={lns_k}, RCL={lns_rcl})")
    print(f"  Start from                   : {start_from}")
    print()

    return best_assignment, Z_best_primal


# =============================================================================
# Point d'entrée
# =============================================================================

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))

    stock_file = os.path.join(base_dir, "Stock_proj_chute.csv")
    plan_file = os.path.join(base_dir, "Plan de production.csv")
    seuil_file = os.path.join(base_dir, "seuil_reincorpo_mini.csv")

    seuil_mini = load_seuil_reincorpo_mini(seuil_file)

    solve_lagrangian_lns(stock_file, plan_file, seuil_mini)
