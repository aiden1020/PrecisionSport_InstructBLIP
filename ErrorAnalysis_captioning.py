from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import json
from collections import Counter


def evaluate_caption_scores(gt_path, result_path):
    coco = COCO(gt_path)
    coco_result = coco.loadRes(result_path)
    coco_eval = COCOEvalCap(coco, coco_result)
    coco_eval.params["image_id"] = coco.getImgIds()
    coco_eval.evaluate()
    return coco_eval


def find_low_score_samples(coco_eval, gt_path, result_path, output_json="low_score_samples.json", threshold_cider=1.0, threshold_bleu4=0.90):
    with open(gt_path, "r") as f:
        gt_data = json.load(f)["annotations"]  
    with open(result_path, "r") as f:
        pred_data = json.load(f)

    gt_dict = {item["image_id"]: item["caption"] for item in gt_data}

    pred_dict = {item["image_id"]: item["caption"] for item in pred_data}

    low_score_samples = []

    for score in coco_eval.evalImgs:
        if score["CIDEr"] < threshold_cider or score["Bleu_4"] < threshold_bleu4:
            image_id = score["image_id"]
            sample = {
                "image_id": image_id,
                "gt_caption     ": gt_dict.get(image_id, "N/A"),
                "predict_caption": pred_dict.get(image_id, "N/A")
            }
            low_score_samples.append(sample)

    with open(output_json, "w") as f:
        json.dump(low_score_samples, f, indent=4)
    return low_score_samples

def analyze_common_errors(gt_path, result_path, low_score_ids):
    with open(gt_path, "r") as f:
        gt_data = json.load(f)["annotations"]  
    with open(result_path, "r") as f:
        pred_data = json.load(f)

    gt_dict = {item["image_id"]: item["caption"] for item in gt_data}

    pred_dict = {item["image_id"]: item["caption"] for item in pred_data}

    error_phrases = Counter()
    for item in low_score_ids:
        img_id = item["image_id"] if isinstance(item, dict) and "image_id" in item else item
        gt_words = gt_dict.get(img_id, "").split()
        pred_words = pred_dict.get(img_id, "").split()  

        gt_phrases = set([" ".join(gt_words[i:i+2]) for i in range(len(gt_words) - 1)])
        pred_phrases = set([" ".join(pred_words[i:i+2]) for i in range(len(pred_words) - 1)])

        missing_phrases = gt_phrases - pred_phrases 
        error_phrases.update(missing_phrases)

    return error_phrases.most_common(20)  

if __name__ == "__main__":
    gt_path = "lavis/configs/datasets/badminton_caption/input/val_gt.json"
    result_path = "output/results/eval/20250504213/result/test_badminton_caption_result_epochbest.json"
    with open(result_path, "r") as f:
        pred_data = json.load(f)

    with open(gt_path, "r") as f:
        gt_data = json.load(f)["annotations"]

    coco_eval = evaluate_caption_scores(gt_path, result_path)
    print(f"CIDEr: {coco_eval.eval['CIDEr']:.4f}, SPICE: {coco_eval.eval['SPICE']:.4f}")

    print(f"預測結果中總共有 {len(pred_data)} 條字幕")
    print(f"Ground Truth 中有 {len(gt_data)} 條標註")
    low_score_ids = find_low_score_samples(coco_eval, gt_path, result_path)
    print(f"低分樣本數量: {len(low_score_ids)} { len(low_score_ids) / len(pred_data) * 100:.2f}%")

    common_errors = analyze_common_errors(gt_path, result_path, low_score_ids)
    print(f"最常出錯的詞: {common_errors}")
