import json
from datasets import load_dataset

dataset = load_dataset("hotpotqa/hotpot_qa", "distractor")

for split, out_file in [("validation", "hotpotqa_dev.json"),
                         ("train",      "hotpotqa_train.json")]:
    records = []
    for ex in dataset[split]:
        records.append({
            "question": ex["question"],
            "answer":   ex["answer"],
            "context":  list(zip(
                            ex["context"]["title"],
                            ex["context"]["sentences"]   # list of sentence-lists
                        )),
        })
    with open(out_file, "w") as f:
        json.dump(records, f)
    print(f"Wrote {len(records)} records to {out_file}")