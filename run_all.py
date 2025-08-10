import os

for opt in [1,2,4]:
    print(F"Running Bohmian option = {opt}")
    os.system(F"python run_cauchy.py --opt={opt}")

