from transformers import AutoTokenizer, AutoModelForCausalLM 

model_name = "/home/RYVEN/weighty-matters/models/Qwen3.5-0.8B" 

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype="auto",
        )

print("Model loaded successfully")
print("Parameters:", model.num_parameters())
