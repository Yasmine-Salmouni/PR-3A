import csv
import itertools
import os
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
# Relaxation Lagrangienne
# =============================================================================

def solve_lagrangian(stock_file, plan_file, seuil_reincorpo_mini=100.0,
                     max_iter=200, alpha_init=2.0, epsilon=1):
    """
    Résout le modèle de ré-incorporation par relaxation lagrangienne.

    Args:
        stock_file:            chemin vers Stock_proj_chute.csv
        plan_file:             chemin vers Plan de production.csv
        seuil_reincorpo_mini:  masse minimale (kg) pour engager une activation
        max_iter:              nombre maximal d'itérations
        alpha_init:            pas initial du sous-gradient
        epsilon:               seuil de convergence sur le gap
    """

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
    print("RELAXATION LAGRANGIENNE — RÉ-INCORPORATION DES CHUTES")
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
    print(f"  Max itérations                      : {max_iter}")
    print(f"  Epsilon (critère d'arrêt)           : {epsilon}")
    print()
    print("Résolution en cours...")
    print()

    best_primal = -float("inf")
    best_R_primal = {}
    best_O_primal = {}
    alpha = alpha_init
    no_improve_count = 0

    # -----------------------------------------------------------------
    # 2. Initialisation des multiplicateurs de Lagrange
    # -----------------------------------------------------------------
    # λ[c, dc] >= 0 pour chaque chute c et fenêtre dc
    lam = {}
    for (c, dc) in stock_proj_chute:
        lam[(c, dc)] = 0.0

    t_start = time.time()

    print(f"  {'Iter':>5s}  |  {'Z_dual':>12s}  |  {'Z_primal':>12s}"
          f"  |  {'Meilleur':>12s}  |  {'Gap':>10s}  |  {'α':>8s}")
    print("  " + "-" * 80)

    for iteration in range(1, max_iter + 1):

        # =============================================================
        # ÉTAPE A : Résolution des sous-problèmes (Borne Duale)
        # =============================================================
        # Pour chaque (p, dp), on cherche la meilleure chute c à
        # ré-incorporer en maximisant le score net.
        #
        # Score(c) = (1 - λ[c,dc]) × reincorpo_maxi[p,dp,c] - 1
        #
        # On sélectionne la chute avec le meilleur score > 0.
        # Sinon, O = 0 et R = 0 pour ce créneau.

        R_A = {}                       # (c, dc, p, dp) -> volume
        O_A = {}                       # (p, dp, c) -> 1
        usage_A = defaultdict(float)   # (c, dc) -> somme des R_A
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
                O_A[(p, dp, c)] = 1
                R_A[(c, dc, p, dp)] = maxi
                usage_A[(c, dc)] += maxi
                sum_best_scores += best_score

        # Z_dual = Σ best_scores + Σ λ[c,dc] × stock[c,dc]
        lambda_stock_sum = sum(
            lam[key] * stock_proj_chute[key] for key in stock_proj_chute
        )
        Z_dual = sum_best_scores + lambda_stock_sum

        # =============================================================
        # ÉTAPE B : Heuristique de réparation (Borne Primale)
        # =============================================================

        # B.1 : Lister et trier les activations par volume décroissant
        activations = []
        for (c, dc, p, dp), vol in R_A.items():
            activations.append((c, dc, p, dp, vol))
        activations.sort(key=lambda x: -x[4])

        # B.2 : Allouer le stock réel
        remaining_stock = dict(stock_proj_chute)
        R_B = {}
        O_B = {}
        used_pdp = set()   # pour unicité (p, dp)

        for (c, dc, p, dp, vol) in activations:
            # B.3 : Unicité — une seule chute par (p, dp)
            if (p, dp) in used_pdp:
                continue

            avail = remaining_stock.get((c, dc), 0.0)
            actual_vol = min(vol, avail)

            # Vérifier le seuil minimal
            if actual_vol >= seuil_reincorpo_mini:
                R_B[(c, dc, p, dp)] = actual_vol
                O_B[(p, dp, c)] = 1
                remaining_stock[(c, dc)] = avail - actual_vol
                used_pdp.add((p, dp))
            # sinon : annuler cette activation (O = 0, R = 0)

        # Borne Primale : Z_primal = Σ R - Σ O
        Z_primal = sum(R_B.values()) - len(O_B)

        # Mise à jour de la meilleure solution primale
        if Z_primal > best_primal:
            best_primal = Z_primal
            best_R_primal = dict(R_B)
            best_O_primal = dict(O_B)
            no_improve_count = 0
        else:
            no_improve_count += 1

        # =============================================================
        # ÉTAPE C : Calcul du Gap
        # =============================================================
        gap = Z_dual - best_primal

        if iteration <= 10 or iteration % 10 == 0 or gap < epsilon:
            print(
                f"  {iteration:5d}  |  {Z_dual:12.2f}  |  {Z_primal:12.2f}"
                f"  |  {best_primal:12.2f}  |  {gap:10.2f}  |  {alpha:8.4f}"
            )

        if gap < epsilon:
            print(f"\n  Convergence atteinte (gap = {gap:.4f} < ε = {epsilon})")
            break

        # =============================================================
        # ÉTAPE D : Mise à jour des multiplicateurs (sous-gradient)
        # =============================================================
        # g[c,dc] = stock[c,dc] - Σ_{p,dp} R_A[c,dc,p,dp]
        # Si surconsommation (g < 0) → λ augmente (chute plus "chère")
        # Si sous-consommation (g > 0) → λ diminue (chute moins "chère")
        g = {}
        sum_g2 = 0.0
        for (c, dc) in stock_proj_chute:
            g[(c, dc)] = stock_proj_chute[(c, dc)] - usage_A.get((c, dc), 0.0)
            sum_g2 += g[(c, dc)] ** 2

        if sum_g2 < 1e-12:
            print(f"\n  Sous-gradient nul à l'itération {iteration}.")
            break

        # Pas du sous-gradient
        step = alpha * (Z_dual - best_primal) / sum_g2

        for (c, dc) in lam:
            lam[(c, dc)] = max(0.0, lam[(c, dc)] - step * g[(c, dc)])

        # Réduction de alpha si pas d'amélioration depuis 30 itérations
        if no_improve_count > 0 and no_improve_count % 30 == 0:
            alpha *= 0.5

    # Message d'arrêt si limite atteinte
    if iteration == max_iter:
        print(f"\n  Arrêt de sécurité après {iteration} itérations (gap = {gap:.4f})")

    t_end = time.time()

    # Statut et temps (comme modèle initial)
    if gap < epsilon:
        status = "OPTIMAL"
    elif best_primal > -float("inf"):
        status = "FEASIBLE"
    else:
        status = "NOT_SOLVED"
    
    print(f"Statut : {status}")
    print(f"Temps de résolution : {t_end - t_start:.3f} secondes\n")

    # -----------------------------------------------------------------
    # 7. Résultats finaux
    # -----------------------------------------------------------------
    total_reincorpore = sum(best_R_primal.values())
    #La somme de toutes les quantités ré-incorporées par le modèle
    total_stock = sum(stock_proj_chute.values())
    #La somme de toutes les quantités de chute disponibles
    nb_activations = len(best_O_primal)
    #Le nombre d'activations

    print()
    print("=" * 70)
    print("RÉSULTATS (meilleure solution primale)")
    print("=" * 70)
    print(f"  Valeur objectif (Z_primal)  : {best_primal:.2f}")
    print(f"  Volume total ré-incorporé   : {total_reincorpore:.2f} kg")
    print(f"  Stock total initial disponible : {total_stock:.2f} kg")
    print(f"  Taux de ré-incorporation (Volume total ré-incorporé/"
          f" Stock total initial disponible) : "
          f"{100 * total_reincorpore / total_stock:.1f} %")
    print(f"  Nombre d'activations        : {nb_activations}")
    print()

    # Détail par chute
    print("-" * 70)
    print("DÉTAIL PAR CHUTE")
    print("-" * 70)
    for c in C:
        vol_c = sum(v for (c2, dc, p, dp), v in best_R_primal.items()
                    if c2 == c)
        stock_c = sum(v for (c2, _), v in stock_proj_chute.items()
                      if c2 == c)
        if vol_c > 0.01:
            print(f"\n  {c}  (stock: {stock_c:.0f} kg "
                  f"→ ré-incorporé: {vol_c:.0f} kg)")
            for key in sorted(best_R_primal):
                if key[0] == c and best_R_primal[key] > 0.01:
                    c_, dc_, p_, dp_ = key
                    dmc, dlc = dc_
                    print(
                        f"    R[{c_}, ({dmc.date()}, {dlc.date()}), "
                        f"{p_}, {dp_.date()}] = {best_R_primal[key]:.2f} kg"
                    )

    # Multiplicateurs de Lagrange finaux
    print("\n" + "-" * 70)
    print("MULTIPLICATEURS DE LAGRANGE FINAUX (λ)")
    print("-" * 70)
    # Regrouper par chute et trier par lambda décroissant
    lam_by_chute = defaultdict(list)
    for (c, dc), val in lam.items():
        dmc, dlc = dc
        lam_by_chute[c].append((dmc, dlc, val))
    for c in sorted(lam_by_chute):
        lam_by_chute[c].sort(key=lambda x: -x[2])  # tri par λ décroissant
        print(f"\n  {c}")
        for dmc, dlc, val in lam_by_chute[c]:
            print(f"    λ[{c}, ({dmc.date()}, {dlc.date()})] = {val:.6f}")
    
    # Statistiques lambda final
    all_lam = list(lam.values())
    lam_min = min(all_lam)
    lam_max = max(all_lam)
    lam_avg = sum(all_lam) / len(all_lam)
    lam_nonzero = sum(1 for v in all_lam if v > 1e-6)
    
    print(f"\nStatistiques λ final :")
    print(f"  Min    : {lam_min:.6f}")
    print(f"  Max    : {lam_max:.6f}")
    print(f"  Moyenne: {lam_avg:.6f}")
    print(f"  Non-zéros: {lam_nonzero}/{len(all_lam)}")
    print()

    # Détail des activations
    print(f"\n{'-' * 70}")
    print("ACTIVATIONS (O = 1)")
    print("-" * 70)
    for key in sorted(best_O_primal):
        p, dp, c = key
        print(f"    O[{p}, {dp.date()}, {c}] = 1")

    print(f"\n{'=' * 70}")
    return best_R_primal, best_O_primal


# =============================================================================
# Point d'entrée
# =============================================================================

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))

    stock_file = os.path.join(base_dir, "Stock_proj_chute.csv")
    plan_file = os.path.join(base_dir, "Plan de production.csv")
    seuil_file = os.path.join(base_dir, "seuil_reincorpo_mini.csv")

    seuil_mini = load_seuil_reincorpo_mini(seuil_file)

    solve_lagrangian(stock_file, plan_file, seuil_mini)
