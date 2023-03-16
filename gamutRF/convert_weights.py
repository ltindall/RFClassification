import torch


#weights_filepath = "gamutRF/model_weights/resnet18_0.02_3.pt"
weights_filepath = "gamutRF/model_weights/resnet18_leesburg_split_0.02_1_current.pt"
checkpoint = torch.load(weights_filepath)
state_dict  = checkpoint["model_state_dict"]

label_dirs= {
    'drone': ['data/gamutrf-birdseye-field-days/leesburg_field_day_2022_06_15/worker1/','data/gamutrf-birdseye-field-days/pdx_field_day_2022_05_26/worker1/gamutrf/'], 
    'wifi_2_4': ['data/gamutrf-pdx/07_21_2022/wifi_2_4/'], 
    'wifi_5': ['data/gamutrf-pdx/07_21_2022/wifi_5/']
}
dataset_idx_to_class = {
    0: "drone",
    1: "wifi_2_4", 
    2: "wifi_5"
}
model_checkpoint_data = ({
    "experiment_name": "leesburg_split", 
    "sample_secs": 0.02, 
    "nfft": 512,
    "label_dirs": label_dirs,
    "dataset_idx_to_class": dataset_idx_to_class,
})
model_checkpoint_data["model_state_dict"] = state_dict

torch.save(model_checkpoint_data, "gamutRF/model_weights/resnet18_leesburg_split_0.02_1_current_new.pt")

checkpoint = torch.load("gamutRF/model_weights/resnet18_leesburg_split_0.02_1_current_new.pt")
print(f"{checkpoint['model_state_dict']=}")
print(f"{checkpoint['experiment_name']=}")
print(f"{checkpoint['sample_secs']=}")
print(f"{checkpoint['nfft']=}")
print(f"{checkpoint['label_dirs']=}")
print(f"{checkpoint['dataset_idx_to_class']=}")
