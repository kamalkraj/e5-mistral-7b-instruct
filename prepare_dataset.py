from collections import defaultdict
from datasets import load_dataset,Dataset,DatasetDict

dataset = load_dataset("snli")

train = defaultdict(dict)

for example in dataset["train"]:
    if example["label"] == 0:
        train[example["premise"]]["positive"] = example["hypothesis"]
    elif example["label"] == 2:
        train[example["premise"]]["negative"] = example["hypothesis"]

train_dataset = []
for key,value in train.items():
    if "negative" in value and "positive" in value:
        train_dataset.append({
            "sentence": key,
            "positive": value["positive"],
            "negative": value["negative"]
        })

data = Dataset.from_list(train_dataset)

train_test_split = data.train_test_split()
test_valid = train_test_split['test'].train_test_split(test_size=0.5)
train_test_valid_dataset = DatasetDict({
    'train': train_test_split['train'],
    'test': test_valid['test'],
    'valid': test_valid['train']})

print(train_test_valid_dataset)

train_test_valid_dataset.save_to_disk("similarity_dataset/")