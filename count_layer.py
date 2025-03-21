from transformers import AutoConfig

model_name = "google/gemma-2-9b-it"
config = AutoConfig.from_pretrained(model_name)

print("Number of layers:", config.num_hidden_layers)  # ✅ Use num_hidden_layers
