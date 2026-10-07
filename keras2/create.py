path = "./keras2/"

keras_num = "68"
keyword = "Conv1D_"

names = [
    # "01_california",
    "02_diabetes",
    # "03_boston",
    "04_dacon_ddarung",
    # "05_kaggle_bike",
    "06_cancer",
    # "07_santander",
    "08_wine",
    # "09_fetch_covtype",
    "10_digits",
    # "11_mnist",
    "12_fashion",
    # "13_cifar10",
    "14_cifar100",
    # "15_man_woman",
    # "16_jena",
]

for name in names:
    filename = f"keras{keras_num}_{keyword}{name}.py"
    open(path + filename, "w").close()
    print(f"{filename} 생성")