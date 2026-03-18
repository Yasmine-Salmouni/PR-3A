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
# Heuristique TABU Search multi-opérateurs (corrigée)
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


def _tabu_key(tag: str, *items) -> tuple:
    """Génère une clé tabou."""
    return (tag,) + tuple(items)


class LRUCache:
    """Cache LRU pour les évaluations de solutions."""
    def __init__(self, max_size: int = 5000):
        self.max_size = max(128, int(max_size))
        self.data = OrderedDict()

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


def tabu_search_multiops_fast(
    start_assignment: dict,
    stock_proj_chute: dict,
    reincorpo_maxi: dict,
    options: dict,
    slots: list,
    lambda_campaign: float,
    rng: np.random.Generator,
    # Paramètres principaux
    tabu_iters: int = 50,
    tabu_tenure: int = 10,
    inactive_pool: int = 15,
    neighbor_samples: int = 18,
    candidate_chutes_topk: int = 3,
    swap_pairs: int = 8,
    reloc_pairs: int = 8,
    cache_size: int = 6000,
    patience: int = 15,
    # Opérateurs
    enable_replace: bool = True,
    enable_swap: bool = True,
    enable_relocation: bool = False,
    # Améliorations
    reactive_tabu: bool = True,
    tabu_tenure_min: int = 5,
    tabu_tenure_max: int = 20,
    revisit_window: int = 10,
    freq_penalty_weight: float = 0.02,
    restart_after: int = 14,
    intensify_after: int = 5,
    intensify_neighbors: int = 10,
    intensify_swap_pairs: int = 6,
) -> dict:
    """
    Recherche tabou multi-opérateurs corrigée pour la relaxation lagrangienne.
    """
    current = dict(start_assignment)
    _, _, best_obj = evaluate_solution(current, stock_proj_chute, reincorpo_maxi, lambda_campaign)
    best_assignment = dict(current)

    # Pré-calculs
    top_candidates_by_slot = {}
    for (p, dp), opts in options.items():
        ordered = [c for c, dc, maxi in sorted(opts, key=lambda x: x[2], reverse=True) if maxi >= 100.0]
        top_candidates_by_slot[(p, dp)] = ordered[:max(1, int(candidate_chutes_topk))]

    slot_rank = {(p, dp): i for i, (p, dp) in enumerate(slots)}
    cache = LRUCache(cache_size)
    tabu = {}
    freq = defaultdict(int)
    seen_iter = {}

    base_tenure = max(1, int(tabu_tenure))
    tenure = base_tenure
    last_improve_it = 0

    def eval_cached(ass: dict) -> tuple:
        """Évaluation avec cache."""
        sig = tuple(ass.get((p, dp, c), None) for (p, dp, c) in ass.keys())
        val = cache.get(sig)
        if val is not None:
            return val
        total, n_act, obj = evaluate_solution(ass, stock_proj_chute, reincorpo_maxi, lambda_campaign)
        val = (total, n_act, obj)
        cache.put(sig, val)
        return val

    def move_score(obj: float, move_key: tuple) -> float:
        """Score avec pénalité de fréquence."""
        if freq_penalty_weight <= 0:
            return obj
        return obj - float(freq_penalty_weight) * freq[move_key]

    def build_pool(ass: dict) -> tuple:
        """Construit le pool de slots pour les mouvements."""
        active_slots = [(p, dp) for (p, dp, c) in ass.keys()]
        inactive_slots = [(p, dp) for (p, dp) in slots if (p, dp) not in active_slots]
        if inactive_pool > 0 and inactive_slots:
            k = min(int(inactive_pool), len(inactive_slots))
            inact_sample = [tuple(slot) for slot in rng.choice(inactive_slots, size=k, replace=False)]
        else:
            inact_sample = []
        return active_slots + inact_sample, active_slots, inactive_slots

    for it in range(1, int(tabu_iters) + 1):
        if it - last_improve_it >= int(max(1, patience)):
            break

        # Purger les tabous expirés
        expired = [k for k, exp in tabu.items() if exp <= it]
        for k in expired:
            del tabu[k]

        pool, active_slots, inactive_slots = build_pool(current)
        if not pool:
            break

        # Intensification après stagnation
        stagnation = it - last_improve_it
        if stagnation >= int(intensify_after):
            sample_size = min(int(intensify_neighbors), len(pool))
            local_swap_pairs = min(int(swap_pairs), int(intensify_swap_pairs))
        else:
            sample_size = min(int(neighbor_samples), len(pool))
            local_swap_pairs = int(swap_pairs)

        best_move_score = -1e30
        best_move_obj = -1e30
        best_move_key = None
        best_move_assignment = None

        # --------------------------------------------------
        # 1) REPLACE
        # --------------------------------------------------
        if enable_replace:
            for (p, dp) in pool[:sample_size]:
                # Récupérer la chute actuelle pour ce slot
                current_c = None
                current_vol = 0
                for (p2, dp2, c), vol in current.items():
                    if p2 == p and dp2 == dp:
                        current_c = c
                        current_vol = vol
                        break
                
                # Options: None (désactiver) ou autres chutes
                options_list = [None] + top_candidates_by_slot.get((p, dp), [])
                for new_c in options_list:
                    if new_c == current_c:
                        continue

                    move_key = _tabu_key("R", slot_rank[(p, dp)], new_c)
                    trial = dict(current)
                    
                    # Supprimer l'ancienne assignation si exists
                    if current_c is not None:
                        del trial[(p, dp, current_c)]
                    
                    # Ajouter la nouvelle assignation si new_c n'est pas None
                    if new_c is not None:
                        # Trouver le meilleur volume pour cette chute
                        best_volume = 0
                        for c2, dc, maxi in options.get((p, dp), []):
                            if c2 == new_c and maxi >= 100.0:
                                stock_avail = stock_proj_chute.get((c2, dc), 0.0)
                                best_volume = min(maxi, stock_avail)
                                break
                        
                        if best_volume > 0:
                            trial[(p, dp, new_c)] = best_volume
                    
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
        if enable_swap and len(pool) >= 2 and local_swap_pairs > 0:
            n_slots = len(pool)
            max_pairs = min(local_swap_pairs, n_slots * (n_slots - 1) // 2)
            attempts = 0
            while attempts < 5 * max_pairs + 5:
                attempts += 1
                i1, i2 = sorted(rng.choice(n_slots, size=2, replace=False).tolist())
                s1, s2 = pool[i1], pool[i2]
                
                # Récupérer les chutes actuelles
                c1 = None
                c2 = None
                vol1 = 0
                vol2 = 0
                
                for (p, dp, c), vol in current.items():
                    if (p, dp) == s1:
                        c1, vol1 = c, vol
                    elif (p, dp) == s2:
                        c2, vol2 = c, vol
                
                if c1 is None and c2 is None:
                    continue
                
                move_key = _tabu_key("S", min(slot_rank[s1], slot_rank[s2]), max(slot_rank[s1], slot_rank[s2]))
                trial = dict(current)
                
                # Supprimer les anciennes assignations
                if c1 is not None:
                    del trial[(s1[0], s1[1], c1)]
                if c2 is not None:
                    del trial[(s2[0], s2[1], c2)]
                
                # Ajouter les nouvelles assignations (si possible)
                if c2 is not None:
                    # Calculer le meilleur volume pour c1 dans s2
                    best_vol = 0
                    for c, dc, maxi in options.get(s2, []):
                        if c == c1 and maxi >= 100.0:
                            stock_avail = stock_proj_chute.get((c, dc), 0.0)
                            best_vol = min(maxi, stock_avail)
                            break
                    if best_vol > 0:
                        trial[(s2[0], s2[1], c1)] = best_vol
                
                if c1 is not None:
                    # Calculer le meilleur volume pour c2 dans s1
                    best_vol = 0
                    for c, dc, maxi in options.get(s1, []):
                        if c == c2 and maxi >= 100.0:
                            stock_avail = stock_proj_chute.get((c, dc), 0.0)
                            best_vol = min(maxi, stock_avail)
                            break
                    if best_vol > 0:
                        trial[(s1[0], s1[1], c2)] = best_vol
                
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
        # Aucun voisin admissible
        # --------------------------------------------------
        if best_move_assignment is None:
            break

        current = best_move_assignment
        freq[best_move_key] += 1
        tabu[best_move_key] = it + max(1, int(tenure))

        sig = tuple(current.get((p, dp, c), None) for (p, dp, c) in current.keys())
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

        # Redémarrage léger depuis la meilleure solution
        if restart_after > 0 and (it - last_improve_it) >= int(restart_after):
            current = dict(best_assignment)
            last_improve_it = it - int(intensify_after)
            if reactive_tabu:
                tenure = max(int(tabu_tenure_min), base_tenure)

    return best_assignment


# =============================================================================
# Relaxation Lagrangienne avec TABU
# =============================================================================

def solve_lagrangian_tabu(stock_file, plan_file, seuil_reincorpo_mini=100.0,
                         alpha_init=0.1, epsilon=0.01, max_iterations=100,
                         # Paramètres TABU
                         tabu_iters=50, tabu_tenure=10, inactive_pool=15,
                         neighbor_samples=18, candidate_chutes_topk=3,
                         swap_pairs=8, reloc_pairs=8, cache_size=6000,
                         patience=15, enable_replace=True, enable_swap=True,
                         enable_relocation=False, reactive_tabu=True,
                         tabu_tenure_min=5, tabu_tenure_max=20,
                         revisit_window=10, freq_penalty_weight=0.02,
                         restart_after=14, intensify_after=5,
                         intensify_neighbors=10, intensify_swap_pairs=6,
                         # Paramètres communs
                         alpha_urgency=10.0, lambda_campaign=1.0, seed=0):
    """
    Résout le modèle de ré-incorporation par relaxation lagrangienne
    avec heuristique TABU pour la réparation.
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
    print("RELAXATION LAGRANGIENNE — HEURISTIQUE TABU")
    print("=" * 70)
    print(f"  Nombre de slots (p, dp)           : {len(slots)}")
    print(f"  Seuil mini (kg)                   : {seuil_reincorpo_mini}")
    print(f"  TABU iterations                  : {tabu_iters}")
    print(f"  TABU tenure                      : {tabu_tenure}")
    print(f"  Opérateurs activés              : R={enable_replace}, S={enable_swap}, M={enable_relocation}")
    print(f"  Alpha urgency                    : {alpha_urgency}")
    print(f"  Lambda campaign                  : {lambda_campaign}")
    print(f"  Seed                             : {seed}")
    print(f"  Epsilon (critère d'arrêt)        : {epsilon}")
    print(f"  Nombre maximum d'itérations       : {max_iterations}")
    print()
    print("Résolution en cours...")
    print()

    # -----------------------------------------------------------------
    # Step 1 & 2 : Solution initiale valide avec GREEDY
    # -----------------------------------------------------------------
    greedy_assignment = greedy_construct(
        seuil_reincorpo_mini, stock_proj_chute, reincorpo_maxi, 
        options, slots, alpha_urgency, lambda_campaign
    )
    
    Z_best_primal = sum(greedy_assignment.values()) - len(greedy_assignment) * lambda_campaign
    best_assignment = dict(greedy_assignment)

    print(f"  Solution initiale GREEDY Z_best_primal : {Z_best_primal:.2f}")

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
        # Step 7 : Heuristique TABU pour réparation (appliquée à chaque itération)
        # =============================================================
        tabu_repair = tabu_search_multiops_fast(
            best_assignment, stock_proj_chute, reincorpo_maxi, options, slots,
            lambda_campaign, rng,
            tabu_iters=tabu_iters, tabu_tenure=tabu_tenure,
            inactive_pool=inactive_pool, neighbor_samples=neighbor_samples,
            candidate_chutes_topk=candidate_chutes_topk, swap_pairs=swap_pairs,
            reloc_pairs=reloc_pairs, cache_size=cache_size, patience=patience,
            enable_replace=enable_replace, enable_swap=enable_swap,
            enable_relocation=enable_relocation, reactive_tabu=reactive_tabu,
            tabu_tenure_min=tabu_tenure_min, tabu_tenure_max=tabu_tenure_max,
            revisit_window=revisit_window, freq_penalty_weight=freq_penalty_weight,
            restart_after=restart_after, intensify_after=intensify_after,
            intensify_neighbors=intensify_neighbors, intensify_swap_pairs=intensify_swap_pairs
        )
        
        tabu_obj = sum(tabu_repair.values()) - len(tabu_repair) * lambda_campaign
        if tabu_obj > Z_best_primal:
            Z_best_primal = tabu_obj
            best_assignment = dict(tabu_repair)

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
    print(f"  Heuristique utilisée        : TABU (iters={tabu_iters}, tenure={tabu_tenure})")
    print(f"  Opérateurs                   : R={enable_replace}, S={enable_swap}, M={enable_relocation}")
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

    solve_lagrangian_tabu(stock_file, plan_file, seuil_mini)
