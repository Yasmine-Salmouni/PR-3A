import csv
import os
import time
from collections import defaultdict
from datetime import datetime

try:
    import pulp
except ImportError:
    raise ImportError("PuLP n'est pas installé. Installez-le avec: pip install pulp")


# =============================================================================
# Chargement des données
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
# Construction
# =============================================================================

def solve_reincorporation(stock_file, plan_file, seuil_reincorpo_mini=100.0):
    """
    Args:
        stock_file:            chemin vers Stock_proj_chute.csv
        plan_file:             chemin vers Plan de production.csv
        seuil_reincorpo_mini:  masse minimale (kg) pour engager une campagne
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
    variables_dynamiques = {
        "stock_proj_chute": stock_proj_chute,
        "reincorpo_maxi": reincorpo_maxi,
    }

    # Ensembles déduits
    C = sorted({c for (c, _) in stock_proj_chute})
    P = sorted({p for (p, _, _) in reincorpo_maxi})
    DP = sorted({dp for (_, dp, _) in reincorpo_maxi})

    print("=" * 70)
    print("MODÈLE DE RÉ-INCORPORATION DES CHUTES (PULP/CBC)")
    print("=" * 70)
    print(f"  Nombre de chutes (|C|)            : {len(C)}")
    print(f"  Nombre de produits (|P|)          : {len(P)}")
    print(f"  Nombre de dates de production (|DP|): {len(DP)}")
    print(f"  Seuil mini (kg)                   : {seuil_reincorpo_mini} kg")
    print()

    # -----------------------------------------------------------------
    # 2. Création du problème PuLP
    # -----------------------------------------------------------------
    prob = pulp.LpProblem("Reincorporation_Chutes", pulp.LpMaximize)

    # -----------------------------------------------------------------
    # 3. Variables de décision
    # -----------------------------------------------------------------

    # R[c, dc, p, dp]  continu ≥ 0
    
    R = {}

    # Index auxiliaires pour accélérer la construction des contraintes
    R_by_c_dc = defaultdict(list)      # (c, dc)   -> list de clés R
    R_by_p_dp_c = defaultdict(list)    # (p, dp, c) -> list de clés R

    for (p, dp, c), maxi in reincorpo_maxi.items():
        if c not in chute_windows:
            continue 
        for dc in chute_windows[c]:
            dmc, dlc = dc
            if dmc <= dp <= dlc: #Contrainte: Respect de la fenêtre d'utilisation de la chute
                ## Créé uniquement si dp ∈ dc  (fenêtre d'utilisation respectée)
                stock_val = stock_proj_chute.get((c, dc), 0)
                upper = min(stock_val, maxi)
                name = f"R_{c}_{dmc.strftime('%m%d')}_{dlc.strftime('%m%d')}_{p}_{dp.strftime('%m%d')}"
                var = pulp.LpVariable(name, lowBound=0, upBound=upper, cat='Continuous')
                key = (c, dc, p, dp)
                R[key] = var
                R_by_c_dc[(c, dc)].append(key)
                R_by_p_dp_c[(p, dp, c)].append(key)

    # O[p, dp, c]  booléen
    O = {}
    for (p, dp, c) in reincorpo_maxi:
        if c not in chute_windows:
            continue
        # Créer O seulement s'il existe au moins une variable R associée
        if (p, dp, c) in R_by_p_dp_c:
            name = f"O_{p}_{dp.strftime('%m%d')}_{c}"
            O[(p, dp, c)] = pulp.LpVariable(name, cat='Binary')

    print(f"  Nombre de variables R créées      : {len(R)}")
    print(f"  Nombre de variables O créées      : {len(O)}")
    print()

    # -----------------------------------------------------------------
    # 4. Contraintes
    # -----------------------------------------------------------------
    nb = {}

    # 4a. Disponibilité du stock
    # ∑_p ∑_dp R[c,dc,p,dp] ≤ stock_proj_chute[c,dc]   ∀c ∈ C, ∀dc ∈ DC
    cnt = 0
    for (c, dc), stock_val in stock_proj_chute.items():
        keys = R_by_c_dc.get((c, dc), [])
        if keys:
            prob += pulp.lpSum([R[k] for k in keys]) <= stock_val, f"stock_{c}_{dc[0].strftime('%m%d')}_{dc[1].strftime('%m%d')}"
            cnt += 1
    nb["stock"] = cnt

    # 4b. Capacité d'absorption du produit
    # ∑_dc R[c,dc,p,dp] ≤ reincorpo_maxi[p,dp,c] × O[p,dp,c]   ∀p,dp,c
    cnt = 0
    for (p, dp, c), maxi in reincorpo_maxi.items():
        if (p, dp, c) not in O:
            continue
        keys = R_by_p_dp_c[(p, dp, c)]
        prob += pulp.lpSum([R[k] for k in keys]) <= maxi * O[(p, dp, c)], f"cap_{p}_{dp.strftime('%m%d')}_{c}"
        cnt += 1
    nb["capacité"] = cnt

    # 4c. Seuil minimal d'engagement
    # seuil × O[p,dp,c] ≤ ∑_dc R[c,dc,p,dp]   ∀p,dp,c
    cnt = 0
    for (p, dp, c) in list(O.keys()):
        keys = R_by_p_dp_c[(p, dp, c)]
        prob += seuil_reincorpo_mini * O[(p, dp, c)] <= pulp.lpSum([R[k] for k in keys]), f"seuil_{p}_{dp.strftime('%m%d')}_{c}"
        cnt += 1
    nb["seuil"] = cnt

    # 4d. Unicité : une seule chute par (produit, date de production)
    # ∑_c O[p,dp,c] ≤ 1   ∀p ∈ P, ∀dp ∈ DP
    cnt = 0
    pd_chutes = defaultdict(list)
    for (p, dp, c) in O:
        pd_chutes[(p, dp)].append(c)

    for (p, dp), chutes in pd_chutes.items():
        prob += pulp.lpSum([O[(p, dp, c)] for c in chutes]) <= 1, f"unique_{p}_{dp.strftime('%m%d')}"
        cnt += 1
    nb["unicité"] = cnt

    print("  Nombre de contraintes ajoutées :")
    for label, n in nb.items():
        print(f"    {label:12s} : {n}")
    print()

    # -----------------------------------------------------------------
    # 5. Fonction objectif
    # -----------------------------------------------------------------
    # Max ∑_{c,dc,p,dp} ( R[c,dc,p,dp] − O[p,dp,c] )
    
    # Calcul des coefficients pour les variables O
    o_coeff = defaultdict(float)
    for (c, dc, p, dp) in R.keys():
        o_coeff[(p, dp, c)] -= 1.0
    
    # Construction de la fonction objectif
    objective_terms = []
    
    # Ajouter les variables R avec coefficient +1
    for var in R.values():
        objective_terms.append(var)
    
    # Ajouter les variables O avec leurs coefficients négatifs
    for key_o, coeff in o_coeff.items():
        if key_o in O:
            objective_terms.append(coeff * O[key_o])
    
    prob += pulp.lpSum(objective_terms), "Objectif"

    # -----------------------------------------------------------------
    # 6. Résolution
    # -----------------------------------------------------------------
    print("Résolution en cours...")
    t_start = time.time()
    
    # Résolution avec le solveur CBC (par défaut dans PuLP)
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    t_end = time.time()

    # Affichage du statut
    status_map = {
        pulp.LpStatusOptimal: "OPTIMAL",
        pulp.LpStatusNotSolved: "NOT_SOLVED",
        pulp.LpStatusInfeasible: "INFEASIBLE",
        pulp.LpStatusUnbounded: "UNBOUNDED",
        pulp.LpStatusUndefined: "UNDEFINED",
    }
    
    print(f"Statut : {status_map.get(pulp.LpStatus[prob.status], pulp.LpStatus[prob.status])}")
    print(f"Temps de résolution : {t_end - t_start:.3f} secondes\n")

    if prob.status not in (pulp.LpStatusOptimal, pulp.LpStatusNotSolved):
        print("Aucune solution trouvée.")
        return None

    # -----------------------------------------------------------------
    # 7. Résultats
    # -----------------------------------------------------------------
    total_reincorpore = sum(var.varValue for var in R.values())
    #La somme de toutes les quantités ré-incorporées par le modèle
    total_stock = sum(stock_proj_chute.values())
    #La somme de toutes les quantités de chute disponibles
    nb_activations = sum(1 for var in O.values() if var.varValue > 0.5)
    #Le nombre d'activations

    print("=" * 70)
    print("RÉSULTATS")
    print("=" * 70)
    print(f"  Valeur objectif          : {pulp.value(prob.objective):.2f}")
    print(f"  Volume total ré-incorporé: {total_reincorpore:.2f} kg")
    print(f"  Stock total initial disponible   : {total_stock:.2f} kg")
    print(f"  Taux de ré-incorporation (Volume total ré-incorporé/ Stock total initial disponible) : {100 * total_reincorpore / total_stock:.1f} %")
    print(f"  Nombre d'activations              : {nb_activations}")
    print()

    # Détail par chute
    # print("-" * 70)
    # print("DÉTAIL PAR CHUTE")
    # print("-" * 70)
    # for c in C:
    #     vol_c = sum(R[k].varValue for k in R if k[0] == c)
    #     stock_c = sum(v for (c2, _), v in stock_proj_chute.items() if c2 == c)
    #     if vol_c > 0.01:
    #         print(f"\n  {c}  (stock: {stock_c:.0f} kg → ré-incorporé: {vol_c:.0f} kg)")
    #         for key in sorted(R):
    #             if key[0] == c and R[key].varValue > 0.01:
    #                 c_, dc_, p_, dp_ = key
    #                 dmc, dlc = dc_
    #                 print(
    #                     f"    R[{c_}, ({dmc.date()}, {dlc.date()}), "
    #                     f"{p_}, {dp_.date()}] = {R[key].varValue:.2f} kg"
    #                 )

    # Détail des activations
    # print(f"\n{'-' * 70}")
    # print("ACTIVATIONS (O = 1)")
    # print("-" * 70)
    # for key in sorted(O):
    #     if O[key].varValue > 0.5:
    #         p, dp, c = key
    #         print(f"    O[{p}, {dp.date()}, {c}] = 1")

    print(f"\n{'=' * 70}")
    return prob


# =============================================================================
# Point d'entrée
# =============================================================================

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))

    stock_file = os.path.join(base_dir, "Stock_proj_chute.csv")
    plan_file = os.path.join(base_dir, "Plan de production.csv")
    seuil_file = os.path.join(base_dir, "seuil_reincorpo_mini.csv")

    seuil_mini = load_seuil_reincorpo_mini(seuil_file)

    solve_reincorporation(stock_file, plan_file, seuil_mini)
