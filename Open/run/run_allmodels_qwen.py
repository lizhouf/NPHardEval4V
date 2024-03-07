import subprocess

# 定义要运行的脚本列表以及每个脚本对应的参数
arg_model = 'fuyu_8b'

scripts_and_args = [
    ("run_p_BSP.py", [arg_model]),
    ("run_p_EDP.py", [arg_model]),
    ("run_hard_GCP.py", [arg_model]),
    ("run_cmp_GCP_D.py", [arg_model]),
    ("run_cmp_KSP.py", [arg_model]),
    ("run_hard_MSP.py", [arg_model]),
    ("run_p_SPP.py", [arg_model]),
    ("run_hard_TSP.py", [arg_model]),
    ("run_cmp_TSP_D.py", [arg_model])
]

# 遍历脚本列表并依次执行
for script, args in scripts_and_args:
    print(f"Running script: {script} with arguments: {args}")
    # 使用subprocess模块运行脚本，并传递参数
    subprocess.run([f"/anaconda/envs/flam_new/bin/python", "-u", script]+ args, check=True)
    print(f"Script {script} finished.")

print("All scripts have finished running.")
