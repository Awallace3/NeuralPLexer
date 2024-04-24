from torchmdnet.models.model import load_model
model_path = "/storage/ice1/7/3/awallace43/torchmd_data/epoch=2139-val_loss=0.2543-test_loss=0.2317.ckpt"

model = load_model(model_path, derivative=True)
print(model)
