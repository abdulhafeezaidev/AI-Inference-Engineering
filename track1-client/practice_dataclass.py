from dataclasses import dataclass
@dataclass
class GeneralConfig:
    token_limit: int
    temperature: float

config = GeneralConfig(
    token_limit=100, temperature=0.5
    )

print(config)
print(config.token_limit)
print(config.temperature)

