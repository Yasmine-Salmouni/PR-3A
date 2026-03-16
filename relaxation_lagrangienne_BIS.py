import csv
import itertools
import os
import signal
import sys
import time
from collections import defaultdict
from datetime import datetime


# =============================================================================
# Chargement des données (identique au modèle initial)
# =============================================================================

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
# Relaxation Lagrangienne BIS (selon l'article Fisher 1985)
# =============================================================================

def solve_lagrangian_bis(stock_file, plan_file, seuil_reincorpo_mini=100.0,
                       alpha_init=0.1, epsilon=0.01, max_iterations=100):
    """
    Résout le modèle de ré-incorporation par relaxation lagrangienne
    selon la méthodologie de Fisher (1985).

    Args:
        stock_file:            chemin vers Stock_proj_chute.csv
        plan_file:             chemin vers Plan de production.csv
        seuil_reincorpo_mini:  masse minimale (kg) pour engager une activation
        alpha_init:            pas initial du sous-gradient (fixé à 2.0)
        epsilon:               seuil de convergence sur le gap
        max_iterations:         nombre maximum d'itérations (300)
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
    # Variables dynamiques (mises à jour quotidiennement)
    # -----------------------------------------------------------------
    # - stock_proj_chute[c, dc] : stock projeté (kg) de la chute c utilisable sur la fenêtre dc
    # - reincorpo_maxi[p, dp, c] : capacité maximale (kg) d'incorporation de la chute c dans le produit p à la date dp

    # Ensembles déduits
    C = sorted({c for (c, _) in stock_proj_chute})
    P = sorted({p for (p, _, _) in reincorpo_maxi})
    DP = sorted({dp for (_, dp, _) in reincorpo_maxi})

    # -----------------------------------------------------------------
    # Pré-calcul : options valides par (p, dp)
    # -----------------------------------------------------------------
    # Pour chaque (p, dp), liste des (c, dc, maxi) utilisables
    # Contrainte : dp ∈ dc (fenêtre d'utilisation respectée)
    # Contrainte : maxi >= seuil (sinon impossible d'atteindre le seuil minimal)
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

    print("=" * 70)
    print("RELAXATION LAGRANGIENNE BIS — MÉTHODE FISHER (1985)")
    print("=" * 70)
    print(f"  Nombre de chutes (|C|)              : {len(C)}")
    print(f"  Nombre de produits (|P|)            : {len(P)}")
    print(f"  Nombre de dates de production (|DP|): {len(DP)}")
    print(f"  Seuil mini (kg)                     : {seuil_reincorpo_mini}")
    
    # Statistiques des variables et contraintes
    nb_R = sum(len(opts) for opts in options.values())  # nombre total d'options (c,dc,p,dp)
    nb_O = sum(len(opts) for opts in options.values())  # même nombre (une activation par option)
    nb_stock = len(stock_proj_chute)                     # une contrainte stock par (c,dc) - RELAXÉE
    nb_capacite = sum(len(opts) for opts in options.values())  # une contrainte capacité par R valide
    nb_seuil = sum(len(opts) for opts in options.values())      # une contrainte seuil par O valide
    nb_unicite = len(options)                            # une contrainte unicité par (p,dp)
    
    print(f"  Nombre de variables R créées      : {nb_R}")
    print(f"  Nombre de variables O créées      : {nb_O}")
    print()
    print(f"  Nombre de contraintes ajoutées :")
    print(f"    stock        : {nb_stock} (relaxée)")
    print(f"    capacité     : {nb_capacite}")
    print(f"    seuil        : {nb_seuil}")
    print(f"    unicité      : {nb_unicite}")
    print()
    print(f"  Nombre de sous-problèmes (p, dp)    : {len(options)}")
    print(f"  Epsilon (critère d'arrêt)           : {epsilon}")
    print(f"  Nombre maximum d'itérations         : {max_iterations}")
    print()
    print("Résolution en cours...")
    print()

    # -----------------------------------------------------------------
    # Step 1 & 2 : Solution initiale valide pour Z_best_primal
    # -----------------------------------------------------------------
    # Créer un plan simple qui respecte les stocks
    Z_best_primal = -float("inf")
    best_R_primal = {}
    best_O_primal = {}
    
    # Plan simple : alimenter seulement les 3 premières productions avec stock suffisant
    remaining_stock = dict(stock_proj_chute)
    used_pdp = set()
    
    simple_plan_R = {}
    simple_plan_O = {}
    
    for (p, dp), opts in sorted(options.items())[:3]:  # 3 premières productions
        best_score = 0.0
        best_choice = None
        
        for (c, dc, maxi) in opts:
            avail = remaining_stock.get((c, dc), 0.0)
            if avail >= seuil_reincorpo_mini and avail >= maxi:
                score = maxi - 1.0  # Score simple sans lambda
                if score > best_score:
                    best_score = score
                    best_choice = (c, dc, maxi)
        
        if best_choice:
            c, dc, maxi = best_choice
            simple_plan_R[(c, dc, p, dp)] = maxi
            simple_plan_O[(p, dp, c)] = 1
            remaining_stock[(c, dc)] -= maxi
            used_pdp.add((p, dp))
    
    # Calculer le gain de ce plan simple
    if simple_plan_R:
        Z_best_primal = sum(simple_plan_R.values()) - len(simple_plan_O)
        best_R_primal = dict(simple_plan_R)
        best_O_primal = dict(simple_plan_O)
    
    print(f"  Solution initiale Z_best_primal : {Z_best_primal:.2f}")

    # -----------------------------------------------------------------
    # Step 3 : Initialisation des multiplicateurs et paramètres
    # -----------------------------------------------------------------
    k = 1
    alpha = alpha_init
    # Initialiser λ à 0.1 comme dans le code de l'auteure
    lam = {}
    for (c, dc) in stock_proj_chute:
        lam[(c, dc)] = 0.1

    # Compteur pour la règle des 5 itérations consécutives
    no_improve_count = 0
    last_Z_dual = -float("inf")
    best_Z_dual_ever = float("inf")  # Record absolu du meilleur Z_dual

    t_start = time.time()

    # Commenté : Tableau des itérations
    # print(f"  {'Iter':>5s}  |  {'Z_dual':>12s}  |  {'Z_primal':>12s}"
    #       f"  |  {'Meilleur':>12s}  |  {'Gap':>10s}  |  {'α':>8s}")
    # print("  " + "-" * 80)

    for iteration in range(1, max_iterations + 1):
        
        # Vérifier l'interruption avant chaque itération
        if interrupted:
            break

        # =============================================================
        # Step 4 & 5 : Résolution des sous-problèmes (Étape A de Score)
        # =============================================================
        # Pour chaque production (p, dp), calcule le Score = [(1 - lambda) * reincorpo_max] - 1
        R_brut = {}                       # (c, dc, p, dp) -> volume
        O_brut = {}                       # (p, dp, c) -> 1
        usage_brut = defaultdict(float)   # (c, dc) -> somme des R_brut
        sum_best_scores = 0.0

        for (p, dp), opts in options.items():
            best_score = 0.0
            best_choice = None

            for (c, dc, maxi) in opts:
                score = (1.0 - lam[(c, dc)]) * maxi - 1.0
                if score > best_score:
                    best_score = score
                    best_choice = (c, dc, maxi)

            if best_choice is not None:
                c, dc, maxi = best_choice
                O_brut[(p, dp, c)] = 1
                R_brut[(c, dc, p, dp)] = maxi
                usage_brut[(c, dc)] += maxi
                sum_best_scores += best_score

        # =============================================================
        # Step 6 : Borne Duale
        # =============================================================
        # Z_dual = Σ (λ × stock) + Σ (Scores Max Positifs)
        lambda_stock_sum = sum(
            lam[key] * stock_proj_chute[key] for key in stock_proj_chute
        )
        Z_dual = lambda_stock_sum + sum_best_scores

        # =============================================================
        # Step 7 : Test de Faisabilité strict + Heuristique de réparation
        # =============================================================
        # Vérifier si Σ R_brut ≤ stock_proj_chute pour toutes les chutes
        feasible = True
        for (c, dc), stock_disponible in stock_proj_chute.items():
            utilisation = usage_brut.get((c, dc), 0.0)
            if utilisation > stock_disponible + 1e-6:  # Tolérance numérique
                feasible = False
                break
        
        if feasible:
            # Solution naturellement valide
            Z_primal = sum(R_brut.values()) - len(O_brut)
            if Z_primal > Z_best_primal:
                Z_best_primal = Z_primal
                best_R_primal = dict(R_brut)
                best_O_primal = dict(O_brut)
        else:
            # Solution non-faisable : appliquer l'heuristique de réparation
            # pour trouver une meilleure valeur Meilleur
            R_repare = {}
            O_repare = {}
            remaining_stock = dict(stock_proj_chute)
            
            # Appliquer le même principe que l'heuristique gloutonne initiale
            for (p, dp), opts in sorted(options.items()):
                best_score = -float("inf")
                best_choice = None
                
                for (c, dc, maxi) in opts:
                    avail = remaining_stock.get((c, dc), 0.0)
                    if avail >= seuil_reincorpo_mini and avail >= maxi:
                        # Score sans lambda (heuristique pure)
                        score = maxi - 1.0
                        if score > best_score:
                            best_score = score
                            best_choice = (c, dc, maxi)
                
                if best_choice:
                    c, dc, maxi = best_choice
                    R_repare[(c, dc, p, dp)] = maxi
                    O_repare[(p, dp, c)] = 1
                    remaining_stock[(c, dc)] -= maxi
            
            # Calculer le score de la solution réparée
            if R_repare:
                Z_primal_repare = sum(R_repare.values()) - len(O_repare)
                if Z_primal_repare > Z_best_primal:
                    Z_best_primal = Z_primal_repare
                    best_R_primal = dict(R_repare)
                    best_O_primal = dict(O_repare)

        # =============================================================
        # Step 8 : Sous-gradient et Mise à jour
        # =============================================================
        
        # Test d'arrêt
        gap = Z_dual - Z_best_primal
        if gap <= epsilon:
            print(f"\n  Convergence atteinte (gap = {gap:.4f} ≤ ε = {epsilon})")
            break

        # Mise à jour de α : règle des 5 itérations consécutives basée sur le record absolu
        if Z_dual < best_Z_dual_ever - 1e-6:  # Nouveau record absolu
            best_Z_dual_ever = Z_dual
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= 5:
                alpha = alpha / 2.0
                no_improve_count = 0
                print(f"  α réduit à {alpha:.4f} après 5 itérations sans battre le record")
        
        last_Z_dual = Z_dual

        # Calcul des erreurs g
        g = {}
        sum_g2 = 0.0
        for (c, dc) in stock_proj_chute:
            g[(c, dc)] = usage_brut.get((c, dc), 0.0) - stock_proj_chute[(c, dc)]
            sum_g2 += g[(c, dc)] ** 2

        if sum_g2 < 1e-12:
            print(f"\n  Sous-gradient nul à l'itération {iteration}.")
            break

        # Calcul du pas t_k
        step = (alpha * (Z_dual - Z_best_primal)) / sum_g2

        # Nouveaux prix λ
        for (c, dc) in lam:
            lam[(c, dc)] = max(0.0, lam[(c, dc)] + step * g[(c, dc)])

        # Commenté : Affichage des itérations
        # if iteration <= 10 or iteration % 10 == 0 or gap < epsilon:
        #     print(
        #         f"  {iteration:5d}  |  {Z_dual:12.2f}  |  {Z_best_primal:12.2f}"
        #         f"  |  {Z_best_primal:12.2f}  |  {gap:10.2f}  |  {alpha:8.4f}"
        #     )
        
        # Vérifier l'interruption après la mise à jour
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
    total_reincorpore = sum(best_R_primal.values())
    total_stock = sum(stock_proj_chute.values())
    nb_activations = len(best_O_primal)
    
    # Calcul des violations de contraintes de stock
    violations_stock = {}
    nb_violations = 0
    total_violation = 0.0
    
    # Calcul de l'utilisation réelle par chute et fenêtre
    usage_reel = defaultdict(float)
    for (c, dc, p, dp), vol in best_R_primal.items():
        usage_reel[(c, dc)] += vol
    
    # Vérification des violations
    for (c, dc), stock_disponible in stock_proj_chute.items():
        utilisation = usage_reel.get((c, dc), 0.0)
        violation = utilisation - stock_disponible
        if violation > 1e-6:  # Seuil de tolérance numérique
            violations_stock[(c, dc)] = violation
            nb_violations += 1
            total_violation += violation

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
    print(f"  Contraintes de stock violées : {nb_violations} / {len(stock_proj_chute)}")
    if nb_violations > 0:
        print(f"  Volume total en violation    : {total_violation:.2f} kg")
    
    if interrupted:
        print(f"\n  Note : Résultats basés sur {iteration-1} itérations complètes")
    
    print()

    # Détail des violations de stock si elles existent
    if nb_violations > 0:
        print("-" * 70)
        print("DÉTAIL DES VIOLATIONS DE CONTRAINTES DE STOCK")
        print("-" * 70)
        for (c, dc), violation in sorted(violations_stock.items(), 
                                        key=lambda x: -x[1]):
            stock_dispo = stock_proj_chute[(c, dc)]
            utilisation = usage_reel.get((c, dc), 0.0)
            print(f"  {c} - {dc[0].strftime('%Y-%m-%d')} au {dc[1].strftime('%Y-%m-%d')}")
            print(f"    Stock disponible : {stock_dispo:.2f} kg")
            print(f"    Utilisation     : {utilisation:.2f} kg")
            print(f"    Violation       : +{violation:.2f} kg")
            print()
        print("-" * 70)
        print()

    # Commenté : Détail par chute
    # print("-" * 70)
    # print("DÉTAIL PAR CHUTE")
    # print("-" * 70)
    # for c in C:
    #     vol_c = sum(v for (c2, dc, p, dp), v in best_R_primal.items()
    #                 if c2 == c)
    #     stock_c = sum(v for (c2, _), v in stock_proj_chute.items()
    #                   if c2 == c)
    #     if vol_c > 0.01:
    #         print(f"\n  {c}  (stock: {stock_c:.0f} kg "
    #               f"→ ré-incorporé: {vol_c:.0f} kg)")
    #         for key in sorted(best_R_primal):
    #             if key[0] == c and best_R_primal[key] > 0.01:
    #                 print(f"    {key[3]}: {best_R_primal[key]:.0f} kg")

    return best_R_primal, best_O_primal, Z_best_primal


# =============================================================================
# Point d'entrée
# =============================================================================

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))

    stock_file = os.path.join(base_dir, "Stock_proj_chute.csv")
    plan_file = os.path.join(base_dir, "Plan de production.csv")
    seuil_file = os.path.join(base_dir, "seuil_reincorpo_mini.csv")

    seuil_mini = load_seuil_reincorpo_mini(seuil_file)

    solve_lagrangian_bis(stock_file, plan_file, seuil_mini)
