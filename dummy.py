annotations = {"1": 1, "2": -1}
print(set([object_id for object_id in annotations.keys() if annotations[object_id] != -1]))