import torch
import torchvision
import model_utils.runner as R
import action_recognization_solver
import model_utils.test_tools as test_tools

r=R.runner()

config=action_recognization_solver.vedio_classify_solver.get_defualt_config()
config["task_name"]="test"
config["epochs"]=1
config["dataset_function"]=test_tools.generate_cifar10_dataset
config["dataset_function_params"]={"data_size":32,"batch_size":16}
config["optimizer_function"]=test_tools.generate_optimizers
config["model_class"]=[test_tools.LeNet]
config["mode"]="group"
config["model_params"]=[{}]
config["mem_use"]=[10000]

task={
"solver":{"class":action_recognization_solver.vedio_classify_solver,"params":{}},
"config":config
}

r.generate_tasks([task])
r.main_loop()
