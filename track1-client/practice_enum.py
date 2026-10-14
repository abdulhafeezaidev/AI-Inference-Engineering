from enum import Enum
class RequestMode(str, Enum):
    SYNC = "sync"
    ASYNC = "async"

mode = RequestMode.SYNC

print(mode)
print(mode.value)

valid_mode = RequestMode("sync")
print(valid_mode)

invalid_mode = RequestMode("sycn")
print(invalid_mode)

