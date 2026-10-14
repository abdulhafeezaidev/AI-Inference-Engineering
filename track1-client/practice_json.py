import json
with open("config.json","r") as file:
    config = json.load(file)

print(config)
print(config["model"])
print(config["max_tokens"])

Njson_text = json.dumps(config, indent=4)
print(json_text)

with open("output.json","w") as file:
    json.dump(config, file, indent=4)
    
