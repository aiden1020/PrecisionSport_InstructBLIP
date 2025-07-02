import re
import pandas as pd
df_pred = pd.read_json('output/results/badminton_caption/badminton_caption_1/20250702224/result/val_badminton_caption_result_epoch15.json')
df_gt = pd.read_json('lavis/configs/datasets/badminton_caption/input/val.json')

df = pd.merge(df_pred, df_gt, on='image_id', suffixes=('_pred', '_gt'))
df = df.rename(columns={'caption_pred': 'pred', 'caption_gt': 'refs'})
print(df.head())
def extract_parts(caption):
    pattern = r'(.+?) hits a (.+?) (?:at|in|on) (?:the )?(.+?)(?:\.|$)'
    m = re.match(pattern, caption.lower())
    if not m:
        return None
    _, stroke, area = m.groups()
    stroke = stroke.replace('-', ' ').strip()
    area   = area.replace('-', ' ').strip()
    return stroke, area

total = len(df)
stroke_correct = 0
area_correct   = 0

for _, row in df.iterrows():
    pred = extract_parts(row["pred"])
    if pred is None:
        continue
    pred_stroke, pred_area = pred

    ref_strokes = set()
    ref_areas   = set()
    parts = extract_parts(row["refs"])
    if parts:
        s, a = parts
        ref_strokes.add(s)
        ref_areas.add(a)

    if pred_stroke in ref_strokes:
        stroke_correct += 1
    if pred_area in ref_areas:
        area_correct += 1

stroke_acc = stroke_correct / total
area_acc   = area_correct   / total

print(f"Stroke acc：{stroke_acc:.2%}")
print(f"Area   acc：{area_acc:.2%}")