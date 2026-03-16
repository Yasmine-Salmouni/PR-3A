import csv
import os
import time
from collections import defaultdict
from datetime import datetime

try:
    from pyomo.environ import *
    from pyomo.opt import SolverFactory, TerminationCondition
except ImportError:
    raise ImportError("Pyomo n'est pas installé. Installez-le avec: pip install pyomo")

try:
    # Vérification de la disponibilité de GLPK
    glpk = SolverFactory('glpk')
    if not glpk.available():
        raise ImportError("GLPK n'est pas disponible. Installez-le avec: conda install -c conda-forge glpk")
except:
    raise ImportError("GLPK n'est pas installé ou n'est pas trouvé. Installez-le avec: conda install -c conda-forge glpk")


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
# Construction du Modèle
# =============================================================================

def solve_reincorporation(stock_file, plan_file, seuil_reincorpo_mini=100.0):
    """
    Args:
        stock_file:            chemin vers Stock_proj_chute.csv
        plan_file:             chemin vers Plan de production.csv
        seuil_reincorpo_mini:  masse minimale (kg) pour engager une campagne
    """

    # 1. Chargement des données
    stock_proj_chute, chute_windows = load_stock_proj_chute(stock_file)
    reincorpo_maxi = load_plan_production(plan_file)

    # Ensembles déduits
    C = sorted({c for (c, _) in stock_proj_chute})
    P = sorted({p for (p, _, _) in reincorpo_maxi})
    DP = sorted({dp for (_, dp, _) in reincorpo_maxi})

    print("=" * 70)
    print("MODÈLE DE RÉ-INCORPORATION DES CHUTES (GLPK)")
    print("=" * 70)
    print(f"  Nombre de chutes (|C|)            : {len(C)}")
    print(f"  Nombre de produits (|P|)          : {len(P)}")
    print(f"  Nombre de dates de production (|DP|): {len(DP)}")
    print(f"  Seuil mini (kg)                   : {seuil_reincorpo_mini} kg\n")

    # 2. Création du modèle Pyomo
    model = ConcreteModel("Reincorporation_Chutes")

    # 3. Variables de décision
    
    # Index auxiliaires pour accélérer la construction des contraintes
    R_by_c_dc = defaultdict(list)
    R_by_p_dp_c = defaultdict(list)

    # Construction des index valides pour R
    R_index = []
    for (p, dp, c), maxi in reincorpo_maxi.items():
        if c not in chute_windows:
            continue 
        for dc in chute_windows[c]:
            dmc, dlc = dc
            if dmc <= dp <= dlc: # Respect de la fenêtre d'utilisation de la chute
                stock_val = stock_proj_chute.get((c, dc), 0)
                upper = min(stock_val, maxi)
                key = (c, dc, p, dp)
                R_index.append((key, upper))
                R_by_c_dc[(c, dc)].append(key)
                R_by_p_dp_c[(p, dp, c)].append(key)

    # Création des variables R
    model.R = Var([key for key, _ in R_index], domain=NonNegativeReals)
    
    # Définir les bornes individuellement
    for key, upper in R_index:
        model.R[key].setlb(0.0)
        model.R[key].setub(upper)

    # O[p, dp, c] booléen
    O_index = []
    for (p, dp, c) in reincorpo_maxi:
        if c not in chute_windows:
            continue
        # Créer O seulement s'il existe au moins une variable R associée
        if (p, dp, c) in R_by_p_dp_c:
            O_index.append((p, dp, c))

    # Création des variables O
    model.O = Var(O_index, domain=Binary)

    print(f"  Nombre de variables R créées      : {len(model.R)}")
    print(f"  Nombre de variables O créées      : {len(model.O)}\n")

    # 4. Contraintes
    nb = {}

    # 4a. Disponibilité du stock
    def stock_constraint_rule(model, c, dmc, dlc):
        dc = (dmc, dlc) # On reconstitue le tuple attendu par tes dictionnaires
        keys = R_by_c_dc.get((c, dc), [])
        if keys:
            return sum(model.R[key] for key in keys) <= stock_proj_chute[(c, dc)]
        else:
            return Constraint.Skip
    
    model.stock_constraint = Constraint([(c, dc) for (c, dc) in stock_proj_chute.keys()], rule=stock_constraint_rule)
    nb["stock"] = len(model.stock_constraint)

    # 4b. Capacité d'absorption du produit
    def capacite_constraint_rule(model, p, dp, c):
        if (p, dp, c) not in O_index:
            return Constraint.Skip
        keys = R_by_p_dp_c[(p, dp, c)]
        return sum(model.R[key] for key in keys) <= reincorpo_maxi[(p, dp, c)] * model.O[(p, dp, c)]
    
    model.capacite_constraint = Constraint(O_index, rule=capacite_constraint_rule)
    nb["capacité"] = len(model.capacite_constraint)

    # 4c. Seuil minimal d'engagement
    def seuil_constraint_rule(model, p, dp, c):
        if (p, dp, c) not in O_index:
            return Constraint.Skip
        keys = R_by_p_dp_c[(p, dp, c)]
        return seuil_reincorpo_mini * model.O[(p, dp, c)] <= sum(model.R[key] for key in keys)
    
    model.seuil_constraint = Constraint(O_index, rule=seuil_constraint_rule)
    nb["seuil"] = len(model.seuil_constraint)

    # 4d. Unicité : une seule chute par (produit, date de production)
    pd_chutes = defaultdict(list)
    for (p, dp, c) in O_index:
        pd_chutes[(p, dp)].append(c)

    def unicite_constraint_rule(model, p, dp):
        chutes = pd_chutes.get((p, dp), [])
        if chutes:
            return sum(model.O[(p, dp, c)] for c in chutes) <= 1
        else:
            return Constraint.Skip
    
    model.unicite_constraint = Constraint([(p, dp) for (p, dp) in pd_chutes.keys()], rule=unicite_constraint_rule)
    nb["unicité"] = len(model.unicite_constraint)

    print("  Nombre de contraintes ajoutées :")
    for label, n in nb.items():
        print(f"    {label:12s} : {n}")
    print()

    # 5. Fonction objectif : Max ∑ R - ∑ O
    def objective_rule(model):
        r_sum = sum(model.R[key] for key, _ in R_index)
        o_sum = sum(model.O[key] for key in O_index)
        return r_sum - o_sum
    
    model.objective = Objective(rule=objective_rule, sense=maximize)

    # 6. Résolution avec GLPK
    print("Résolution en cours...")
    t_start = time.time()
    
    solver = SolverFactory('glpk')
    results = solver.solve(model, tee=False)
    t_end = time.time()

    # Vérification robuste du statut Pyomo
    term_cond = results.solver.termination_condition
    print(f"Statut : {term_cond}")
    print(f"Temps de résolution : {t_end - t_start:.3f} secondes\n")

    if term_cond not in [TerminationCondition.optimal, TerminationCondition.feasible]:
        print("Aucune solution trouvée ou problème irréalisable.")
        return None

    # 7. Résultats
    total_reincorpore = sum(value(model.R[key]) for key, _ in R_index)
    total_stock = sum(stock_proj_chute.values())
    nb_activations = sum(1 for key in O_index if value(model.O[key]) > 0.5)

    print("=" * 70)
    print("RÉSULTATS")
    print("=" * 70)
    print(f"  Valeur objectif          : {value(model.objective):.2f}")
    print(f"  Volume total ré-incorporé: {total_reincorpore:.2f} kg")
    print(f"  Stock total initial      : {total_stock:.2f} kg")
    taux = (100 * total_reincorpore / total_stock) if total_stock > 0 else 0
    print(f"  Taux de ré-incorporation : {taux:.1f} %")
    print(f"  Nombre d'activations     : {nb_activations}\n")

    # print("-" * 70)
    # print("DÉTAIL PAR CHUTE")
    # print("-" * 70)
    # for c in C:
    #     vol_c = sum(value(model.R[key]) for key, _ in R_index if key[0] == c)
    #     stock_c = sum(v for (c2, _), v in stock_proj_chute.items() if c2 == c)
    #     if vol_c > 0.01:
    #         print(f"\n  {c}  (stock: {stock_c:.0f} kg → ré-incorporé: {vol_c:.0f} kg)")
    #         for key, _ in sorted(R_index):
    #             if key[0] == c and value(model.R[key]) > 0.01:
    #                 c_, dc_, p_, dp_ = key
    #                 dmc, dlc = dc_
    #                 print(f"    R[{c_}, ({dmc.date()}, {dlc.date()}), {p_}, {dp_.date()}] = {value(model.R[key]):.2f} kg")

    # print(f"\n{'-' * 70}")
    # print("ACTIVATIONS (O = 1)")
    # print("-" * 70)
    # for key in sorted(O_index):
    #     if value(model.O[key]) > 0.5:
    #         p, dp, c = key
    #         print(f"    O[{p}, {dp.date()}, {c}] = 1")

    print(f"\n{'=' * 70}")
    return model


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))

    stock_file = os.path.join(base_dir, "Stock_proj_chute.csv")
    plan_file = os.path.join(base_dir, "Plan de production.csv")
    seuil_file = os.path.join(base_dir, "seuil_reincorpo_mini.csv")

    seuil_mini = load_seuil_reincorpo_mini(seuil_file)
    solve_reincorporation(stock_file, plan_file, seuil_mini)