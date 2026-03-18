import csv
import os
import time
from collections import defaultdict
from datetime import datetime

try:
    import cplex
except ImportError:
    raise ImportError("Cplex n'est pas installé. Installez-le avec: pip install cplex")


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
    print("MODÈLE DE RÉ-INCORPORATION DES CHUTES (CPLEX)")
    print("=" * 70)
    print(f"  Nombre de chutes (|C|)            : {len(C)}")
    print(f"  Nombre de produits (|P|)          : {len(P)}")
    print(f"  Nombre de dates de production (|DP|): {len(DP)}")
    print(f"  Seuil mini (kg)                   : {seuil_reincorpo_mini} kg")
    print()

    # -----------------------------------------------------------------
    # 2. Création du solveur Cplex
    # -----------------------------------------------------------------
    problem = cplex.Cplex()
    
    # Configuration du solveur
    problem.set_problem_name("Reincorporation_Chutes")
    problem.set_problem_type(cplex.Cplex.problem_type.MILP)
    
    # Paramètres pour limiter la sortie et optimiser
    problem.set_log_stream(None)  # Désactiver les logs détaillés
    problem.set_error_stream(None)  # Désactiver les erreurs détaillées
    problem.set_warning_stream(None)  # Désactiver les warnings
    problem.set_results_stream(None)  # Désactiver les résultats détaillés
    
    # Limiter le temps de calcul si nécessaire (optionnel)
    # problem.set_timelimit(300)  # 5 minutes max

    # -----------------------------------------------------------------
    # 3. Variables de décision
    # -----------------------------------------------------------------

    # R[c, dc, p, dp]  continu ≥ 0
    
    R_vars = []  # Liste des noms de variables R
    R_keys = []  # Liste des clés correspondantes
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
                var_name = f"R_{c}_{dmc.strftime('%m%d')}_{dlc.strftime('%m%d')}_{p}_{dp.strftime('%m%d')}"
                key = (c, dc, p, dp)
                
                # Ajout de la variable continue
                problem.variables.add(names=[var_name], lb=[0.0], ub=[upper])
                
                R_vars.append(var_name)
                R_keys.append(key)
                R_by_c_dc[(c, dc)].append(key)
                R_by_p_dp_c[(p, dp, c)].append(key)

    # O[p, dp, c]  booléen
    O_vars = []  # Liste des noms de variables O
    O_keys = []  # Liste des clés correspondantes
    
    for (p, dp, c) in reincorpo_maxi:
        if c not in chute_windows:
            continue
        # Créer O seulement s'il existe au moins une variable R associée
        if (p, dp, c) in R_by_p_dp_c:
            var_name = f"O_{p}_{dp.strftime('%m%d')}_{c}"
            key = (p, dp, c)
            
            # Ajout de la variable binaire
            problem.variables.add(names=[var_name], types=[problem.variables.type.binary])
            
            O_vars.append(var_name)
            O_keys.append(key)

    # Dictionnaires pour accéder facilement aux variables
    R_dict = dict(zip(R_keys, R_vars))
    O_dict = dict(zip(O_keys, O_vars))

    print(f"  Nombre de variables R créées      : {len(R_vars)}")
    print(f"  Nombre de variables O créées      : {len(O_vars)}")
    print()

    # -----------------------------------------------------------------
    # 4. Contraintes
    # -----------------------------------------------------------------
    constraints = {}
    constraint_names = []

    # 4a. Disponibilité du stock
    # ∑_p ∑_dp R[c,dc,p,dp] ≤ stock_proj_chute[c,dc]   ∀c ∈ C, ∀dc ∈ DC
    cnt = 0
    for (c, dc), stock_val in stock_proj_chute.items():
        keys = R_by_c_dc.get((c, dc), [])
        if keys:
            var_names = [R_dict[k] for k in keys]
            coeff = [1.0] * len(var_names)
            constraint_name = f"stock_{c}_{dc[0].strftime('%m%d')}_{dc[1].strftime('%m%d')}"
            
            problem.linear_constraints.add(
                lin_expr=[cplex.SparsePair(ind=var_names, val=coeff)],
                senses=["L"],
                rhs=[stock_val],
                names=[constraint_name]
            )
            
            constraint_names.append(constraint_name)
            cnt += 1
    constraints["stock"] = cnt

    # 4b. Capacité d'absorption du produit
    # ∑_dc R[c,dc,p,dp] ≤ reincorpo_maxi[p,dp,c] × O[p,dp,c]   ∀p,dp,c
    cnt = 0
    for (p, dp, c), maxi in reincorpo_maxi.items():
        if (p, dp, c) not in O_dict:
            continue
        keys = R_by_p_dp_c[(p, dp, c)]
        var_names = [R_dict[k] for k in keys]
        
        # Variables dans la partie gauche : R variables
        # Variables dans la partie droite : O variable (déplacée à gauche avec coefficient négatif)
        all_vars = var_names + [O_dict[(p, dp, c)]]
        coeffs = [1.0] * len(var_names) + [-maxi]
        
        constraint_name = f"cap_{p}_{dp.strftime('%m%d')}_{c}"
        
        problem.linear_constraints.add(
            lin_expr=[cplex.SparsePair(ind=all_vars, val=coeffs)],
            senses=["L"],
            rhs=[0.0],
            names=[constraint_name]
        )
        
        constraint_names.append(constraint_name)
        cnt += 1
    constraints["capacité"] = cnt

    # 4c. Seuil minimal d'engagement
    # seuil × O[p,dp,c] ≤ ∑_dc R[c,dc,p,dp]   ∀p,dp,c
    cnt = 0
    for (p, dp, c) in O_keys:
        keys = R_by_p_dp_c[(p, dp, c)]
        var_names = [R_dict[k] for k in keys]
        
        # Variables : R variables - O variable
        all_vars = var_names + [O_dict[(p, dp, c)]]
        coeffs = [1.0] * len(var_names) + [-seuil_reincorpo_mini]
        
        constraint_name = f"seuil_{p}_{dp.strftime('%m%d')}_{c}"
        
        problem.linear_constraints.add(
            lin_expr=[cplex.SparsePair(ind=all_vars, val=coeffs)],
            senses=["G"],
            rhs=[0.0],
            names=[constraint_name]
        )
        
        constraint_names.append(constraint_name)
        cnt += 1
    constraints["seuil"] = cnt

    # 4d. Unicité : une seule chute par (produit, date de production)
    # ∑_c O[p,dp,c] ≤ 1   ∀p ∈ P, ∀dp ∈ DP
    cnt = 0
    pd_chutes = defaultdict(list)
    for (p, dp, c) in O_keys:
        pd_chutes[(p, dp)].append(c)

    for (p, dp), chutes in pd_chutes.items():
        var_names = [O_dict[(p, dp, c)] for c in chutes]
        coeff = [1.0] * len(var_names)
        constraint_name = f"unique_{p}_{dp.strftime('%m%d')}"
        
        problem.linear_constraints.add(
            lin_expr=[cplex.SparsePair(ind=var_names, val=coeff)],
            senses=["L"],
            rhs=[1.0],
            names=[constraint_name]
        )
        
        constraint_names.append(constraint_name)
        cnt += 1
    constraints["unicité"] = cnt

    print("  Nombre de contraintes ajoutées :")
    for label, n in constraints.items():
        print(f"    {label:12s} : {n}")
    print()

    # -----------------------------------------------------------------
    # 5. Fonction objectif
    # -----------------------------------------------------------------
    # Max ∑_{c,dc,p,dp} ( R[c,dc,p,dp] − O[p,dp,c] )
    
    # Coefficients pour les variables R
    obj_coeffs_R = [1.0] * len(R_vars)
    
    # Coefficients pour les variables O
    o_coeff = defaultdict(float)
    for (c, dc, p, dp) in R_keys:
        o_coeff[(p, dp, c)] -= 1.0
    
    obj_coeffs_O = []
    for key in O_keys:
        obj_coeffs_O.append(o_coeff.get(key, 0.0))
    
    # Combiner toutes les variables et leurs coefficients
    all_vars = R_vars + O_vars
    all_coeffs = obj_coeffs_R + obj_coeffs_O
    
    # Définir l'objectif
    problem.objective.set_linear(list(zip(all_vars, all_coeffs)))
    problem.objective.set_sense(problem.objective.sense.maximize)

    # -----------------------------------------------------------------
    # 6. Résolution
    # -----------------------------------------------------------------
    print("Résolution en cours...")
    t_start = time.time()
    
    try:
        problem.solve()
        t_end = time.time()
        
        status = problem.solution.get_status()
        status_map = {
            problem.solution.status.optimal: "OPTIMAL",
            problem.solution.status.feasible: "FEASIBLE",
            problem.solution.status.infeasible: "INFEASIBLE",
            problem.solution.status.unbounded: "UNBOUNDED",
            problem.solution.status.not_solved: "NOT_SOLVED",
        }
        
        print(f"Statut : {status_map.get(status, status)}")
        print(f"Temps de résolution : {t_end - t_start:.3f} secondes\n")
        
        if status not in (problem.solution.status.optimal, problem.solution.status.feasible):
            print("Aucune solution trouvée.")
            return None
            
    except cplex.exceptions.CplexError as e:
        print(f"Erreur lors de la résolution: {e}")
        return None

    # -----------------------------------------------------------------
    # 7. Résultats
    # -----------------------------------------------------------------
    
    # Récupérer les valeurs des variables
    R_values = problem.solution.get_values(R_vars)
    O_values = problem.solution.get_values(O_vars)
    
    # Créer des dictionnaires pour faciliter l'accès
    R_solution = dict(zip(R_keys, R_values))
    O_solution = dict(zip(O_keys, O_values))
    
    total_reincorpore = sum(R_values)
    #La somme de toutes les quantités ré-incorporées par le modèle
    total_stock = sum(stock_proj_chute.values())
    #La somme de toutes les quantités de chute disponibles
    nb_activations = sum(1 for v in O_values if v > 0.5)
    #Le nombre d'activations

    print("=" * 70)
    print("RÉSULTATS")
    print("=" * 70)
    print(f"  Valeur objectif          : {problem.solution.get_objective_value():.2f}")
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
    #     vol_c = sum(R_solution[k] for k in R_keys if k[0] == c)
    #     stock_c = sum(v for (c2, _), v in stock_proj_chute.items() if c2 == c)
    #     if vol_c > 0.01:
    #         print(f"\n  {c}  (stock: {stock_c:.0f} kg → ré-incorporé: {vol_c:.0f} kg)")
    #         for key in sorted(R_keys):
    #             if key[0] == c and R_solution[key] > 0.01:
    #                 c_, dc_, p_, dp_ = key
    #                 dmc, dlc = dc_
    #                 print(
    #                     f"    R[{c_}, ({dmc.date()}, {dlc.date()}), "
    #                     f"{p_}, {dp_.date()}] = {R_solution[key]:.2f} kg"
    #                 )

    # Détail des activations
    # print(f"\n{'-' * 70}")
    # print("ACTIVATIONS (O = 1)")
    # print("-" * 70)
    # for key in sorted(O_keys):
    #     if O_solution[key] > 0.5:
    #         p, dp, c = key
    #         print(f"    O[{p}, {dp.date()}, {c}] = 1")

    print(f"\n{'=' * 70}")
    return problem


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
