import subprocess

# 定义要运行的脚本列表以及每个脚本对应的参数
scripts_and_args = [
    ("run_p_BSP.py", ["kosmos2"]),
    ("run_p_EDP.py", ["kosmos2"]),
    ("run_hard_GCP.py", ["kosmos2"]),
    ("run_cmp_GCP_D.py", ["kosmos2"]),
    ("run_cmp_KSP.py", ["kosmos2"]),
    ("run_hard_MSP.py", ["kosmos2"]),
    ("run_p_SPP.py", ["kosmos2"]),
    ("run_hard_TSP.py", ["kosmos2"]),
    ("run_cmp_TSP_D.py", ["kosmos2"])
]

# 遍历脚本列表并依次执行
for script, args in scripts_and_args:
    print(f"Running script: {script} with arguments: {args}")
    # 使用subprocess模块运行脚本，并传递参数
    subprocess.run([f"/home/lixiang/.conda/envs/nphardeval4v/bin/python", "-u", script] + args, check=True)
    print(f"Script {script} finished.")

print("All scripts have finished running.")
