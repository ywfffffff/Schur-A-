from transformers import AutoModelForCausalLM, AutoTokenizer
import inspect

model = AutoModelForCausalLM.from_pretrained("/home/lab1008/data_disk_sdc/ywf/models/Qwen/Qwen3-30B-A3B-Instruct-2507", trust_remote_code=False, device_map="auto")
print("model class:", model.__class__)
print("python file:", inspect.getfile(model.__class__))
print("forward defined in:", inspect.getsourcefile(model.forward))