import json
import os
import statistics
import sys
from typing import List, Dict, Any, Optional


def compute_metrics(records: List[Dict[str, Any]]) -> Dict[str, float]:
	"""Compute aggregate metrics for a list of records (no grouping)."""
	return _compute_metrics_core(records)


def _compute_metrics_core(records: List[Dict[str, Any]]) -> Dict[str, float]:
	# Split answerable vs impossible (answerable determined by non-empty gt_list)
	answerable = [r for r in records if r.get("gt_list")]
	impossible = [r for r in records if not r.get("gt_list")]

	hit1_count = 0
	exact_match_count = 0
	precisions: List[float] = []
	recalls: List[float] = []
	f1s: List[float] = []

	for r in answerable:
		pred_list = r.get("pred_list") or []
		gt_list = r.get("gt_list") or []
		pred_set = set(pred_list)
		gt_set = set(gt_list)
		if pred_list and pred_list[0] in gt_set:
			hit1_count += 1
		if pred_set == gt_set:
			exact_match_count += 1
		precision = len(pred_set & gt_set) / len(pred_set) if pred_list else 0.0
		recall = len(pred_set & gt_set) / len(gt_set) if gt_list else 0.0
		f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
		precisions.append(precision)
		recalls.append(recall)
		f1s.append(f1)

	total_ans = len(answerable)
	metrics_ans = {
		"hit@1": hit1_count / total_ans * 100 if total_ans else 0.0,
		"exact_match": exact_match_count / total_ans * 100 if total_ans else 0.0,
		"precision": statistics.mean(precisions) * 100 if precisions else 0.0,
		"recall": statistics.mean(recalls) * 100 if recalls else 0.0,
		"f1": statistics.mean(f1s) * 100 if f1s else 0.0,
	}

	impossible_correct = sum(1 for r in impossible if not r.get("pred_list"))
	total_imp = len(impossible)
	metrics_imp = {
		"impossible_accuracy": impossible_correct / total_imp * 100 if total_imp else 0.0,
	}

	return {
		**metrics_ans,
		**metrics_imp,
		"agg_metrics": metrics_ans["f1"],
		"total_answerable": total_ans,
		"total_impossible": total_imp,
		"total_records": len(records),
	}


def compute_grouped_metrics(records: List[Dict[str, Any]], group_key: str) -> Dict[str, Dict[str, float]]:
	"""Compute metrics per group (e.g., per question_type).

	Returns mapping: group_value -> metrics dict.
	Records missing the group_key are grouped under "__MISSING__".
	"""
	groups: Dict[str, List[Dict[str, Any]]] = {}
	for r in records:
		key = r.get(group_key)
		if key is None:
			key = "__MISSING__"
		groups.setdefault(key, []).append(r)
	grouped_metrics = {k: _compute_metrics_core(v) for k, v in sorted(groups.items())}
	return grouped_metrics


def main():
	if len(sys.argv) < 2:
		print("Usage: python eval_qa_type.py <result_json_path> [--group question_type]")
		sys.exit(1)
	path = sys.argv[1]
	group_key: Optional[str] = None
	if len(sys.argv) > 2:
		# Simple arg parse: expect --group <key>
		if sys.argv[2] == "--group" and len(sys.argv) > 3:
			group_key = sys.argv[3]
		else:
			print("Unrecognized arguments. Use --group <field> if grouping is desired.")
			sys.exit(1)
	if not os.path.isfile(path):
		print(f"File not found: {path}")
		sys.exit(1)
	with open(path, "r", encoding="utf-8") as f:
		records = json.load(f)

	output: Dict[str, Any] = {"overall": compute_metrics(records)}
	if group_key:
		output["group_key"] = group_key
		output["by_group"] = compute_grouped_metrics(records, group_key)
	print(json.dumps(output, indent=2))


if __name__ == "__main__":
	main()

