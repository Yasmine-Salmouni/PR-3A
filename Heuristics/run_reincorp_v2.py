import time
import pandas as pd

from reincorp_heuristic_v2 import (
    load_data,
    greedy_construct,
    evaluate_solution,
    grasp,
    grasp_tabu,
    grasp_lns,
)

HAS_LNS = True

EXCEL_PATH = "Dataset_1month_anonymized.xlsx"

# ---- paramètres communs
LAMBDA_CAMPAIGN = 1
ALPHA_URGENCY = 10.0

# ---- GRASP
GRASP_ITERS = 200
RCL = 3
RCL_BIAS = 0.70
N_SEEDS_GRASP = 5

# ---- TABU amélioré
TABU_START_FROM = "greedy"
GRASP_TABU_ITERS = 6
TABU_ITERS = 50
TABU_TENURE = 10
INACTIVE_POOL = 15
NEIGHBOR_SAMPLES = 18
CANDIDATE_CHUTES_TOPK = 3
SWAP_PAIRS = 8
RELOC_PAIRS = 8
CACHE_SIZE = 6000
PATIENCE = 15

# ---- améliorations TABU
REACTIVE_TABU = True
TABU_TENURE_MIN = 5
TABU_TENURE_MAX = 20
REVISIT_WINDOW = 10
FREQ_PENALTY_WEIGHT = 0.02
RESTART_AFTER = 14
INTENSIFY_AFTER = 5
INTENSIFY_NEIGHBORS = 10
INTENSIFY_SWAP_PAIRS = 6

# ---- LNS autonome (optionnel)
LNS_START_FROM = "greedy"
GRASP_LNS_ITERS = 10
LNS_ITERS = 80
LNS_K = 15
LNS_RCL = 4
LNS_RCL_BIAS = 0.80
LNS_ALPHA_URGENCY = 10.0
LNS_ACCEPT_NON_IMPROVING = False


TABU_VARIANTS = [
    ("TABU [replace]", dict(enable_replace=True, enable_swap=False, enable_relocation=False)),
    ("TABU [replace+swap]", dict(enable_replace=True, enable_swap=True, enable_relocation=False)),
    ("TABU [replace+swap+reloc]", dict(enable_replace=True, enable_swap=True, enable_relocation=True)),
    ("TABU [replace+reloc]", dict(enable_replace=True, enable_swap=False, enable_relocation=True)),
]


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


def safe_filename(name: str) -> str:
    return name.replace(" ", "_").replace("[", "").replace("]", "").replace("+", "_").replace("/", "_")


def run_tabu_variant(name, threshold, batches_by_chute, cap_by_slot, slots, seed, ops):
    t0 = time.perf_counter()
    total_t, n_camp_t, obj_t, plan_t = grasp_tabu(
        threshold, batches_by_chute, cap_by_slot, slots,
        iterations=GRASP_TABU_ITERS,
        rcl=RCL,
        alpha_urgency=ALPHA_URGENCY,
        lambda_campaign=LAMBDA_CAMPAIGN,
        seed=seed,
        rcl_bias=RCL_BIAS,
        tabu_iters=TABU_ITERS,
        tabu_tenure=TABU_TENURE,
        start_from=TABU_START_FROM,
        inactive_pool=INACTIVE_POOL,
        neighbor_samples=NEIGHBOR_SAMPLES,
        candidate_chutes_topk=CANDIDATE_CHUTES_TOPK,
        swap_pairs=SWAP_PAIRS,
        reloc_pairs=RELOC_PAIRS,
        cache_size=CACHE_SIZE,
        patience=PATIENCE,
        reactive_tabu=REACTIVE_TABU,
        tabu_tenure_min=TABU_TENURE_MIN,
        tabu_tenure_max=TABU_TENURE_MAX,
        revisit_window=REVISIT_WINDOW,
        freq_penalty_weight=FREQ_PENALTY_WEIGHT,
        restart_after=RESTART_AFTER,
        intensify_after=INTENSIFY_AFTER,
        intensify_neighbors=INTENSIFY_NEIGHBORS,
        intensify_swap_pairs=INTENSIFY_SWAP_PAIRS,
        **ops,
    )
    sec = time.perf_counter() - t0
    _, used_t = export_plan(plan_t, f"plan_{safe_filename(name)}.csv")
    return total_t, n_camp_t, obj_t, plan_t, used_t, sec


def run_lns_variant(name, threshold, batches_by_chute, cap_by_slot, slots, seed, start_from):
    t0 = time.perf_counter()
    total_l, n_camp_l, obj_l, plan_l = grasp_lns(
        threshold, batches_by_chute, cap_by_slot, slots,
        iterations=GRASP_LNS_ITERS,
        rcl=RCL,
        alpha_urgency=ALPHA_URGENCY,
        lambda_campaign=LAMBDA_CAMPAIGN,
        seed=seed,
        rcl_bias=RCL_BIAS,
        start_from=start_from,
        lns_iters=LNS_ITERS,
        lns_k=LNS_K,
        lns_rcl=LNS_RCL,
        lns_rcl_bias=LNS_RCL_BIAS,
        lns_alpha_urgency=LNS_ALPHA_URGENCY,
        patience=PATIENCE,
        candidate_chutes_topk=CANDIDATE_CHUTES_TOPK,
        accept_non_improving=LNS_ACCEPT_NON_IMPROVING,
    )
    sec = time.perf_counter() - t0
    _, used_l = export_plan(plan_l, f"plan_{safe_filename(name)}.csv")
    return total_l, n_camp_l, obj_l, plan_l, used_l, sec


def main():
    threshold, batches_by_chute, cap_by_slot, slots = load_data(EXCEL_PATH)

    print("Fichier:", EXCEL_PATH)
    print("Nombre de slots:", len(slots))
    print("Seuil minimal:", threshold)
    print("Lambda campagne:", LAMBDA_CAMPAIGN)
    print("Alpha urgency:", ALPHA_URGENCY)

    results = {}

    # GREEDY
    t0 = time.perf_counter()
    assignment_g = greedy_construct(
        threshold, batches_by_chute, cap_by_slot, slots,
        alpha_urgency=ALPHA_URGENCY,
        lambda_campaign=LAMBDA_CAMPAIGN,
    )
    total_g, n_camp_g, obj_g, plan_g = evaluate_solution(
        assignment_g, threshold, batches_by_chute, cap_by_slot, slots,
        lambda_campaign=LAMBDA_CAMPAIGN,
    )
    sec = time.perf_counter() - t0
    _, used_g = export_plan(plan_g, "plan_greedy.csv")
    print_block("GREEDY", total_g, n_camp_g, obj_g, used_g, len(slots), sec)
    results["GREEDY"] = (total_g, n_camp_g, obj_g, sec)

    # GRASP
    t0 = time.perf_counter()
    best = None
    best_obj = -1e30
    best_seed = None
    for s in range(N_SEEDS_GRASP):
        total_r, n_camp_r, obj_r, plan_r = grasp(
            threshold, batches_by_chute, cap_by_slot, slots,
            iterations=GRASP_ITERS,
            rcl=RCL,
            alpha_urgency=ALPHA_URGENCY,
            lambda_campaign=LAMBDA_CAMPAIGN,
            seed=s,
            rcl_bias=RCL_BIAS,
        )
        if obj_r > best_obj:
            best_obj = obj_r
            best_seed = s
            best = (total_r, n_camp_r, obj_r, plan_r)
    total_r, n_camp_r, obj_r, plan_r = best
    sec = time.perf_counter() - t0
    _, used_r = export_plan(plan_r, "plan_grasp.csv")
    print_block(f"GRASP (best of seeds={best_seed})", total_r, n_camp_r, obj_r, used_r, len(slots), sec)
    results["GRASP"] = (total_r, n_camp_r, obj_r, sec)

    # TABU amélioré
    best_tabu_name = None
    best_tabu_obj = -1e30
    for name, ops in TABU_VARIANTS:
        total_t, n_camp_t, obj_t, plan_t, used_t, sec = run_tabu_variant(
            name, threshold, batches_by_chute, cap_by_slot, slots, seed=0, ops=ops
        )
        print_block(name, total_t, n_camp_t, obj_t, used_t, len(slots), sec)
        results[name] = (total_t, n_camp_t, obj_t, sec)
        if obj_t > best_tabu_obj:
            best_tabu_obj = obj_t
            best_tabu_name = name

    # LNS autonome (optionnel)
    best_lns_name = None
    if HAS_LNS:
        lns_variants = [
            ("LNS [start=greedy]", "greedy"),
            ("LNS [start=grasp]", "grasp"),
        ]

        best_lns_obj = -1e30
        for name, start_from in lns_variants:
            total_l, n_camp_l, obj_l, plan_l, used_l, sec = run_lns_variant(
                name, threshold, batches_by_chute, cap_by_slot, slots, seed=0, start_from=start_from
            )
            print_block(name, total_l, n_camp_l, obj_l, used_l, len(slots), sec)
            results[name] = (total_l, n_camp_l, obj_l, sec)
            if obj_l > best_lns_obj:
                best_lns_obj = obj_l
                best_lns_name = name
    else:
        print("\n[LNS ignoré] grasp_lns non disponible dans l'environnement courant.")

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
    if HAS_LNS:
        print("Meilleure variante LNS:", best_lns_name)
    print("Fichiers générés:")
    print(" - plan_greedy.csv")
    print(" - plan_grasp.csv")
    print(" - plan_TABU_*.csv")
    if HAS_LNS:
        print(" - plan_LNS_*.csv")


if __name__ == "__main__":
    main()
