from reincorp_heuristic import (
    load_data,
    greedy_construct,
    evaluate_solution,
    grasp,
    grasp_tabu
)
import pandas as pd
import time

EXCEL_PATH = "Dataset_45days_30chutes_20products.xlsx"

# ---- paramètres communs
LAMBDA_CAMPAIGN = 1
ALPHA_URGENCY = 10.0

# ---- GRASP
GRASP_ITERS = 200
RCL = 3
RCL_BIAS = 0.70
N_SEEDS_GRASP = 10

# ---- TABU
TABU_START_FROM = "greedy"
GRASP_TABU_ITERS = 10
TABU_ITERS = 80
TABU_TENURE = 15


def export_plan(plan, filename):
    df = pd.DataFrame(plan, columns=["Date", "Produit", "Chute", "Reincorp_kg"])
    df.to_csv(filename, index=False, encoding="utf-8")
    used = int((df["Reincorp_kg"] > 0).sum())
    return df, used


def print_block(name, total, n_camp, obj, used, nslots, sec):
    print(f"\n=== {name} ===")
    print("Volume total (kg):", round(total, 2))
    print("Nb campagnes:", n_camp)
    print("Objectif:", round(obj, 2))
    print(f"Slots utilisés (Reincorp_kg>0): {used} / {nslots}")
    print(f"Temps d'exécution: {sec:.3f} s")


def main():
    threshold, batches_by_chute, cap_by_slot, slots = load_data(EXCEL_PATH)

    print("Fichier:", EXCEL_PATH)
    print("Nombre de slots:", len(slots))
    print("Seuil minimal:", threshold)
    print("Lambda campagne:", LAMBDA_CAMPAIGN)
    print("Alpha urgency:", ALPHA_URGENCY)

    results = {}

    # =======================
    # GREEDY
    # =======================
    t0 = time.perf_counter()
    assignment_g = greedy_construct(
        threshold, batches_by_chute, cap_by_slot, slots,
        alpha_urgency=ALPHA_URGENCY,
        lambda_campaign=LAMBDA_CAMPAIGN
    )
    total_g, n_camp_g, obj_g, plan_g = evaluate_solution(
        assignment_g, threshold, batches_by_chute, cap_by_slot, slots,
        lambda_campaign=LAMBDA_CAMPAIGN
    )
    sec = time.perf_counter() - t0
    _, used_g = export_plan(plan_g, "plan_greedy.csv")
    print_block("GREEDY", total_g, n_camp_g, obj_g, used_g, len(slots), sec)
    results["GREEDY"] = (total_g, n_camp_g, obj_g, sec)

    # =======================
    # GRASP (multi-seed)
    # =======================
    t0 = time.perf_counter()
    best = None
    best_obj = -1e30
    for s in range(N_SEEDS_GRASP):
        total_r, n_camp_r, obj_r, plan_r = grasp(
            threshold, batches_by_chute, cap_by_slot, slots,
            iterations=GRASP_ITERS,
            rcl=RCL,
            alpha_urgency=ALPHA_URGENCY,
            lambda_campaign=LAMBDA_CAMPAIGN,
            seed=s,
            rcl_bias=RCL_BIAS
        )
        if obj_r > best_obj:
            best_obj = obj_r
            best = (total_r, n_camp_r, obj_r, plan_r)
    total_r, n_camp_r, obj_r, plan_r = best
    sec = time.perf_counter() - t0
    _, used_r = export_plan(plan_r, "plan_grasp.csv")
    print_block("GRASP (best of seeds)", total_r, n_camp_r, obj_r, used_r, len(slots), sec)
    results["GRASP"] = (total_r, n_camp_r, obj_r, sec)

    # =======================
    # TABU - tests opérateurs
    # =======================
    combos = [
        ("TABU [replace]", dict(enable_replace=True, enable_swap=False, enable_relocation=False, enable_ruin_recreate=False)),
        ("TABU [replace+swap]", dict(enable_replace=True, enable_swap=True, enable_relocation=False, enable_ruin_recreate=False)),
        ("TABU [replace+swap+reloc]", dict(enable_replace=True, enable_swap=True, enable_relocation=True, enable_ruin_recreate=False)),
        ("TABU [replace+swap+reloc+ruin]", dict(enable_replace=True, enable_swap=True, enable_relocation=True, enable_ruin_recreate=True)),
        ("TABU [replace+reloc]", dict(enable_replace=True, enable_swap=False, enable_relocation=True, enable_ruin_recreate=False)),
    ]

    best_tabu_name = None
    best_tabu = None
    best_tabu_obj = -1e30

    for name, ops in combos:
        t0 = time.perf_counter()
        total_t, n_camp_t, obj_t, plan_t = grasp_tabu(
            threshold, batches_by_chute, cap_by_slot, slots,
            iterations=GRASP_TABU_ITERS,
            rcl=RCL,
            alpha_urgency=ALPHA_URGENCY,
            lambda_campaign=LAMBDA_CAMPAIGN,
            seed=0,
            rcl_bias=RCL_BIAS,
            tabu_iters=TABU_ITERS,
            tabu_tenure=TABU_TENURE,
            start_from=TABU_START_FROM,
            **ops
        )
        sec = time.perf_counter() - t0

        safe_name = name.replace(" ", "_").replace("[", "").replace("]", "").replace("+", "_").replace("/", "_")
        _, used_t = export_plan(plan_t, f"plan_{safe_name}.csv")
        print_block(name, total_t, n_camp_t, obj_t, used_t, len(slots), sec)

        results[name] = (total_t, n_camp_t, obj_t, sec)

        if obj_t > best_tabu_obj:
            best_tabu_obj = obj_t
            best_tabu_name = name
            best_tabu = (total_t, n_camp_t, obj_t, sec)

    # =======================
    # Comparaisons
    # =======================
    print("\n=== COMPARAISONS vs GREEDY ===")
    for k in results:
        if k == "GREEDY":
            continue
        dv = results[k][0] - results["GREEDY"][0]
        dc = results[k][1] - results["GREEDY"][1]
        dobj = results[k][2] - results["GREEDY"][2]
        dt = results[k][3] - results["GREEDY"][3]
        print(f"{k}: Δvol={dv:.2f} kg | Δcamp={dc} | Δobj={dobj:.2f} | Δtemps={dt:.3f} s")

    print("\nMeilleure variante Tabu:", best_tabu_name)
    print("Fichiers générés:")
    print(" - plan_greedy.csv")
    print(" - plan_grasp.csv")
    print(" - plan_TABU_*.csv")


if __name__ == "__main__":
    main()