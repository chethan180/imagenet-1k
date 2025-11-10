import re

log_file_path = "training_best.log"
output_file_path = "metrics.md"

pattern = re.compile(
    r"Step\s+\d+:\s*\{[^}]*'epoch':\s*(?P<epoch>\d+),[^}]*?"
    r"'train_loss':\s*(?P<train_loss>[\d\.eE+-]+),[^}]*?"
    r"'train_top1':\s*(?P<train_top1>[\d\.eE+-]+),[^}]*?"
    r"'train_top5':\s*(?P<train_top5>[\d\.eE+-]+),[^}]*?"
    r"'val_loss':\s*(?P<val_loss>[\d\.eE+-]+),[^}]*?"
    r"'val_top1':\s*(?P<val_top1>[\d\.eE+-]+),[^}]*?"
    r"'val_top5':\s*(?P<val_top5>[\d\.eE+-]+)"
)

rows = []

with open(log_file_path, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        match = pattern.search(line)
        if match:
            rows.append(match.groupdict())

with open(output_file_path, "w", encoding="utf-8") as out:
    out.write("| Epoch | Train Loss | Val Loss | Train Top1 | Val Top1 | Train Top5 | Val Top5 |\n")
    out.write("|------:|-----------:|---------:|-----------:|---------:|-----------:|---------:|\n")
    for r in rows:
        out.write(
            f"| {r['epoch']} | {r['train_loss']} | {r['val_loss']} "
            f"| {r['train_top1']} | {r['val_top1']} | {r['train_top5']} | {r['val_top5']} |\n"
        )

print(f"✅ Markdown file generated at: {output_file_path}")
