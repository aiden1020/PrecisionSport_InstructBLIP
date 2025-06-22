"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging
import json
import os
import torch
import lavis.common.dist_utils as dist_utils
from lavis.common.registry import registry
from lavis.common.vqa_tools.vqa import VQA
from lavis.common.vqa_tools.vqa_vizwiz import VQA_Vizwiz
from lavis.common.vqa_tools.vqa_eval import VQAEval
from lavis.common.vqa_tools.vqa_eval_vizwiz import VQAEval_Vizwiz
from lavis.tasks.base_task import BaseTask
import numpy as np
import re
import pandas as pd
import statistics

@registry.register_task("vqa")
class VQATask(BaseTask):
    def __init__(
        self,
        num_beams,
        max_len,
        min_len,
        evaluate,
        num_ans_candidates,
        inference_method="rank",
        prompt="",
    ):
        super().__init__()

        self.num_beams = num_beams
        self.max_len = max_len
        self.min_len = min_len

        self.evaluate = evaluate
        self.inference_method = inference_method
        self.num_ans_candidates = num_ans_candidates
        self.prompt = prompt

        self.answer_list = None

        self.ques_files = dict()
        self.anno_files = dict()

    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg

        num_beams = run_cfg.get("num_beams", 3)
        max_len = run_cfg.get("max_len", 10)
        min_len = run_cfg.get("min_len", 1)

        evaluate = run_cfg.get("evaluate", False)

        inference_method = run_cfg.get("inference_method", "rank")
        num_ans_candidates = run_cfg.get("num_ans_candidates", 128)
        prompt = run_cfg.get("prompt", "")

        return cls(
            num_beams=num_beams,
            max_len=max_len,
            min_len=min_len,
            evaluate=evaluate,
            num_ans_candidates=num_ans_candidates,
            inference_method=inference_method,
            prompt=prompt,
        )

    def build_datasets(self, cfg):
        datasets = super().build_datasets(cfg)

        # get question file, annotation file and anwser list in COCO format
        for dataset in datasets.values():
            for split in dataset:
                if (
                    hasattr(dataset[split], "coco_fmt_qust_file")
                    and dataset[split].coco_fmt_qust_file is not None
                ):
                    self.ques_files[split] = dataset[split].coco_fmt_qust_file
                    self.anno_files[split] = dataset[split].coco_fmt_anno_file

                try:
                    self.answer_list = dataset[split].answer_list
                except AttributeError:
                    # if answer_list is not provided, then set it to None
                    pass

        if len(self.ques_files) > 0:
            assert len(self.ques_files) == len(
                self.anno_files
            ), "Only support one split for evaluation."

        return datasets

    def valid_step(self, model, samples):
        answers = model.predict_answers(
            samples=samples,
            answer_list=self.answer_list,
            inference_method=self.inference_method,
            num_beams=self.num_beams,
            max_len=self.max_len,
            min_len=self.min_len,
            num_ans_candidates=self.num_ans_candidates,
            prompt=self.prompt,
        )
        pred_qa_pairs = []

        question_id = samples["question_id"]
        for answer, ques_id in zip(answers, question_id):
            ques_id = int(ques_id.item())
            pred_qa_pairs.append({"question_id": ques_id, "answer": answer})

        return pred_qa_pairs

    def after_evaluation(self, val_result, split_name, **kwargs):
        result_file = self.save_result(
            val_result,
            result_dir=registry.get_path("result_dir"),
            filename=f"{split_name}_vqa_result",
            remove_duplicate="question_id",
        )

        metrics = self._report_metrics(result_file=result_file, split=split_name)

        return metrics

    @dist_utils.main_process
    def _report_metrics(self, result_file, split):
        """
        Use official VQA evaluation script to report metrics.
        """
        metrics = {}

        if split in self.ques_files and split in self.anno_files:
            vqa = VQA(self.anno_files[split], self.ques_files[split])
            vqa_result = vqa.loadRes(
                resFile=result_file, quesFile=self.ques_files[split]
            )

            # create vqaEval object by taking vqa and vqaRes
            # n is precision of accuracy (number of places after decimal), default is 2
            vqa_scorer = VQAEval(vqa, vqa_result, n=2)
            logging.info("Start VQA evaluation.")
            vqa_scorer.evaluate()

            # print accuracies
            overall_acc = vqa_scorer.accuracy["overall"]
            metrics["agg_metrics"] = overall_acc

            logging.info("Overall Accuracy is: %.02f\n" % overall_acc)
            logging.info("Per Answer Type Accuracy is the following:")

            for ans_type in vqa_scorer.accuracy["perAnswerType"]:
                logging.info(
                    "%s : %.02f"
                    % (ans_type, vqa_scorer.accuracy["perAnswerType"][ans_type])
                )
                metrics[ans_type] = vqa_scorer.accuracy["perAnswerType"][ans_type]

            with open(
                os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
            ) as f:
                f.write(json.dumps(metrics) + "\n")

        return metrics

@registry.register_task("gqa")
class GQATask(VQATask):
    def valid_step(self, model, samples):
        answers = model.predict_answers(
            samples=samples,
            answer_list=self.answer_list,
            inference_method=self.inference_method,
            num_beams=self.num_beams,
            max_len=self.max_len,
            min_len=self.min_len,
            num_ans_candidates=self.num_ans_candidates,
            prompt=self.prompt,
        )
        pred_qa_pairs = []

        question_id = samples["question_id"]
        gt_answers = samples["answer"]
        
        for answer, ques_id, gt_answer in zip(answers, question_id, gt_answers):
            ques_id = int(ques_id.item())
            pred_qa_pairs.append({"question_id": ques_id, "pred_ans": answer, "gt_ans": gt_answer})

        return pred_qa_pairs
        
    @dist_utils.main_process
    def _report_metrics(self, result_file, split):
        """
        TODO: add other evaluation metrics for GQA
        """

        results = json.load(open(result_file, "r"))
        acc = []
        vqa_tool = VQAEval()

        for res in results:
            if res["gt_ans"] is None:
                # prepare test results for leaderboard evaluation
                self._save_result_leaderboard(results)
                return

            gt_ans = res["gt_ans"]
            pred = res["pred_ans"]

            # if self.inference_method == "generate":
            pred = vqa_tool.processPunctuation(pred)
            pred = vqa_tool.processDigitArticle(pred)

            vqa_acc = 1 if pred == gt_ans else 0

            acc.append(vqa_acc)

        accuracy = sum(acc) / len(acc) * 100
        metrics = {"agg_metrics": accuracy, "acc": accuracy}

        with open(
            os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
        ) as f:
            f.write(json.dumps(metrics) + "\n")

        logging.info(metrics)

        return metrics
        

@registry.register_task("aok_vqa")
class AOKVQATask(VQATask):
    def valid_step(self, model, samples):
        answers = model.predict_answers(
            samples=samples,
            answer_list=self.answer_list,
            inference_method=self.inference_method,
            num_beams=self.num_beams,
            max_len=self.max_len,
            min_len=self.min_len,
            num_ans_candidates=self.num_ans_candidates,
        )

        pred_qa_pairs = []

        question_id = samples["question_id"]
        gt_answers = samples["direct_answers"]

        for pred_answer, ques_id, gt_answer in zip(answers, question_id, gt_answers):
            pred_qa_pairs.append(
                {"question_id": ques_id, "pred_ans": pred_answer, "gt_ans": gt_answer}
            )

        return pred_qa_pairs

    @dist_utils.main_process
    def _report_metrics(self, result_file, split):
        """
        Implementing accuracy computation for AOKVQA, see
        https://github.com/allenai/aokvqa/blob/main/evaluation/eval_predictions.py#L45 for details.
        """
        # TODO add evaluation for multi-choice

        results = json.load(open(result_file, "r"))
        acc = []

        for res in results:
            if res["gt_ans"] is None:
                # prepare test results for leaderboard evaluation
                self._save_result_leaderboard(results)
                return

            pred = res["pred_ans"]
            gt_ans = res["gt_ans"]

            num_match = sum([pred == gt for gt in gt_ans])
            vqa_acc = min(1.0, num_match / 3.0)

            acc.append(vqa_acc)

        accuracy = sum(acc) / len(acc) * 100
        metrics = {"agg_metrics": accuracy, "acc": accuracy}

        with open(
            os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
        ) as f:
            f.write(json.dumps(metrics) + "\n")

        logging.info(metrics)

        return metrics

    @dist_utils.main_process
    def _save_result_leaderboard(self, results):
        """
        Saving the results in the format required for leaderboard evaluation.

        [TODO] add support for multi-choice.
        """
        result_leaderboard = dict()
        for res in results:
            result_leaderboard[res["question_id"]] = {
                "direct_answer": res["pred_ans"],
                "multiple_choice": "",
            }

        result_file = registry.get_path("result_dir") + "_leaderboard.json"

        with open(result_file, "w") as f:
            json.dump(result_leaderboard, f)

        logging.info(f"Saved results for leaderboard evaluation at {result_file}")

@registry.register_task("scienceqa")
class ScienceQATask(VQATask):

    def build_datasets(self, cfg):
        datasets = super().build_datasets(cfg)

        # get question file, annotation file and anwser list in COCO format
        for dataset in datasets.values():
            for split in dataset:
                try:
                    self.answer_list = dataset[split].answer_list
                except AttributeError:
                    # if answer_list is not provided, then set it to None
                    pass

        if len(self.ques_files) > 0:
            assert len(self.ques_files) == len(
                self.anno_files
            ), "Only support one split for evaluation."

        return datasets

    def valid_step(self, model, samples):
        # make predicted answers
        # answers = model.predict_answers(
        #     samples=samples,
        #     answer_list=self.answer_list,
        #     inference_method=self.inference_method,
        #     num_beams=self.num_beams,
        #     max_len=self.max_len,
        #     min_len=self.min_len,
        #     num_ans_candidates=self.num_ans_candidates,
        #     prompt=self.prompt,
        # )
        candidates = []
        if not isinstance(samples, list):
            i = 0
            for choice in samples["choices"][0]:
                label = chr(ord('a') + i)
                candidates.append(f"({label}) {choice}")
                i += 1
        else:
            candidates = samples['choices']
        
        answers = model.predict_class(
            samples=samples,
            candidates=candidates,
            n_segments=1,
        )
        pred_qa_pairs = []


        question_id = samples["question_id"]
        gt_answers = samples["answer"]
        # img_names = samples["image_name"]
        for pred_answer, ques_id, gt_answer in zip(answers, question_id, gt_answers):
            # ques_id = int(ques_id)
            pred_qa_pairs.append({"question_id": ques_id, "pred_ans": pred_answer, "gt_ans": gt_answer})

        return pred_qa_pairs

    def after_evaluation(self, val_result, split_name, **kwargs):
        # print(val_result[:5])
        result_file = self.save_result(
            val_result,
            result_dir=registry.get_path("result_dir"),
            filename=f"{split_name}_scienceqa_result",
            remove_duplicate="",
        )
        if split_name == 'val':
            metrics = self._report_metrics(result_file=result_file, split=split_name)
        else:
            metrics = None 
        return metrics

    @dist_utils.main_process
    def _report_metrics(self, result_file, split):
        # scienceQA metric is easy
        # just check if the predicted answer is the ground truth answer

        results = json.load(open(result_file, "r"))
        acc = []

        for res in results:
            # if res["gt_ans"] is None:
            #     # prepare test results for leaderboard evaluation
            #     self._save_result_leaderboard(results)
            #     return

            pred = res["pred_ans"]
            gt_ans = [res["gt_ans"]]

            num_match = sum([pred == gt for gt in gt_ans])
            vqa_acc = min(1.0, num_match / len(gt_ans))

            acc.append(vqa_acc)

        accuracy = sum(acc) / len(acc) * 100
        metrics = {"agg_metrics": accuracy, "acc": accuracy}

        with open(
            os.path.join(registry.get_path("output_dir"), f"log.txt"), "a"
        ) as f:
            f.write(json.dumps(metrics) + "\n")

        logging.info(metrics)

        return metrics

# @registry.register_task("badminton_qa")
# class BadmintonQATask(VQATask):
#     """
#     Task for chunk-based badminton QA using predict_answers.
#     Eval metrics: Hit@1, Exact Match, token-level F1.
#     """

#     def build_datasets(self, cfg):
#         return super().build_datasets(cfg)

#     # def valid_step(self, model, samples):
#     #     predictions = model.predict_answers(
#     #         samples=samples,
#     #         num_beams=self.num_beams,
#     #         inference_method="generate",
#     #         max_len=self.max_len,
#     #         min_len=self.min_len,
#     #         prompt=self.prompt,
#     #     )

#     #     results = []
#     #     qids = samples["question_id"]
#     #     gts  = samples["answer"]
#     #     for pred, qid, gt in zip(predictions, qids, gts):
#     #         results.append({
#     #             "question_id": qid,
#     #             "pred_ans":    pred.strip(),
#     #             "gt_ans":      gt.strip(),
#     #         })
#     #     return results

#     def extract_ans(text: str) -> str:
#         """
#         從帶 CoT 的文字中抽出最終 '<answer>…</answer>' 或標準句型，
#         若都不符合則回傳空字串。
#         """
#         # 1. 優先取 <answer>…</answer>
#         m = re.search(r"<answer>(.*?)</answer>", text, flags=re.DOTALL)
#         if m:
#             return m.group(1).strip()
#         # 2. 嘗試取標準句型
#         m2 = re.search(r"The event happens at strokes? ([\d,]+)", text)
#         if m2:
#             nums   = m2.group(1)
#             s_or_p = "strokes" if "," in nums else "stroke"
#             return f"The event happens at {s_or_p} {nums}"
#         # 3. 都不符就回空
#         return ""

#     def valid_step(self, model, samples):
#         predictions = model.predict_answers(
#             samples=samples,
#             num_beams=self.num_beams,
#             inference_method="generate",
#             max_len=self.max_len,
#             min_len=self.min_len,
#             prompt=self.prompt,
#         )

#         results = []
#         qids = samples["question_id"]
#         gts  = samples["answer"]
#         for pred, qid, gt in zip(predictions, qids, gts):
#             full_pred = pred.strip()
#             full_gt   = gt.strip()

#             # 抽出預測與標準的純答案
#             pred_ans = extract_ans(full_pred)
#             gt_ans   = extract_ans(full_gt)

#             results.append({
#                 "question_id": qid,
#                 "raw_pred":    full_pred,
#                 "raw_gt":      full_gt,
#                 "pred_ans":    pred_ans,
#                 "gt_ans":      gt_ans,
#             })
#         return results

#     def after_evaluation(self, val_result, split_name,epoch, **kwargs):
#         result_file = self.save_result(
#             val_result,
#             result_dir=registry.get_path("result_dir"),
#             filename=f"{split_name}_badminton_qa_result_epoch{epoch}",
#             remove_duplicate=""
#         )
#         metrics = None
#         if split_name == 'val':
#             metrics = self._report_metrics(result_file=result_file, split=split_name)
#         return metrics

#     @dist_utils.main_process
#     def _report_metrics(self, result_file, split):
#         with open(result_file, "r", encoding="utf-8") as f:
#             records = json.load(f)

#         # step 1: 提取 pred_list 和 gt_list
#         def extract_numbers_per_entry(data):
#             pattern = re.compile(r'\b\d+\b')
#             results = []

#             for entry in data:
#                 pred_numbers = list(set(map(int, pattern.findall(entry.get("pred_ans", "")))))
#                 gt_numbers = list(set(map(int, pattern.findall(entry.get("gt_ans", "")))))
#                 results.append({
#                     "question_id": entry["question_id"],
#                     "pred_list": sorted(pred_numbers),
#                     "gt_list": sorted(gt_numbers)
#                 })

#             return results

#         # step 2: 計算指標
#         def evaluate_predictions(df, hit_k=1):
#             hit1_count = 0
#             exact_match_count = 0
#             all_precisions = []
#             all_recalls = []
#             all_f1s = []

#             for _, row in df.iterrows():
#                 pred = row["pred_list"]
#                 gt = row["gt_list"]

#                 pred_set = set(pred)
#                 gt_set = set(gt)

#                 # Hit@1: 第一個預測是否在 GT 裡
#                 if pred and pred[0] in gt_set:
#                     hit1_count += 1

#                 # Exact Match
#                 if pred_set == gt_set:
#                     exact_match_count += 1

#                 # Precision / Recall / F1
#                 if pred_set:
#                     precision = len(pred_set & gt_set) / len(pred_set)
#                 else:
#                     precision = 1.0 if not gt_set else 0.0

#                 if gt_set:
#                     recall = len(pred_set & gt_set) / len(gt_set)
#                 else:
#                     recall = 1.0 if not pred_set else 0.0

#                 if precision + recall > 0:
#                     f1 = 2 * precision * recall / (precision + recall)
#                 else:
#                     f1 = 0.0

#                 all_precisions.append(precision)
#                 all_recalls.append(recall)
#                 all_f1s.append(f1)

#             total = len(df)
#             if total == 0:
#                 return {
#                     "hit@1": 0.0,
#                     "exact_match": 0.0,
#                     "f1": 0.0,
#                     "agg_metrics": 0.0,
#                 }

#             return {
#                 "hit@1": float(hit1_count / total * 100),
#                 "exact_match": float(exact_match_count / total * 100),
#                 "precision": float(np.mean(all_precisions) * 100),
#                 "recall": float(np.mean(all_recalls) * 100),
#                 "f1": float(np.mean(all_f1s) * 100),
#                 "agg_metrics": float(np.mean(all_f1s) * 100),
#             }

#         # 執行提取與評估
#         processed = extract_numbers_per_entry(records)
#         df = pd.DataFrame(processed)
#         metrics = evaluate_predictions(df)

#         # Logging
#         log_path = os.path.join(registry.get_path("output_dir"), "log.txt")
#         with open(log_path, "a") as f:
#             f.write(json.dumps(metrics) + "\n")
#         logging.info(f"[BadmintonQA] {split} metrics: {metrics}")
#         return metrics

@registry.register_task("badminton_qa")
class BadmintonQATask(VQATask):
    """
    Task for chunk-based badminton QA using predict_answers.
    Eval metrics: Hit@1, Exact Match, token-level F1.
    """

    def build_datasets(self, cfg):
        return super().build_datasets(cfg)

    @staticmethod
    def extract_ans(text: str) -> list[int]:
        """
        從帶 CoT 的文字中抽出最終答案的數字列表；
        若檢測到 <answer>…</answer> 或標準句型，回傳其中的整數；
        否則回傳空 list。
        """
        # 1. 優先取 <answer>…</answer> 裡的數字
        m = re.search(r"<answer>(.*?)</answer>", text, flags=re.DOTALL)
        if m:
            nums = re.findall(r"\b\d+\b", m.group(1))
            return list(map(int, nums))

        # 2. 嘗試取標準句型裡的數字
        m2 = re.search(r"The event happens at strokes? ([\d,]+)", text)
        if m2:
            nums = m2.group(1).split(",")
            return [int(n) for n in nums]

        # 3. 都不符合就回空
        return []

    def valid_step(self, model, samples):
        predictions = model.predict_answers(
            samples=samples,
            num_beams=self.num_beams,
            inference_method="generate",
            max_len=self.max_len,
            min_len=self.min_len,
            prompt=self.prompt,
        )

        results = []
        qids = samples["question_id"]
        gts  = samples["answer"]
        for pred, qid, gt in zip(predictions, qids, gts):
            full_pred = pred.strip()
            full_gt   = gt.strip()

            # 直接從 CoT 中抽出數字列表
            pred_list = self.extract_ans(full_pred)
            gt_list   = self.extract_ans(full_gt)

            results.append({
                "question_id": qid,
                "raw_pred":    full_pred,
                "raw_gt":      full_gt,
                "pred_list":   pred_list,
                "gt_list":     gt_list,
            })
        return results

    def after_evaluation(self, val_result, split_name, epoch, **kwargs):
        result_file = self.save_result(
            val_result,
            result_dir=registry.get_path("result_dir"),
            filename=f"{split_name}_badminton_qa_result_epoch{epoch}",
            remove_duplicate=""
        )
        metrics = None
        if split_name == 'val':
            metrics = self._report_metrics(result_file=result_file, split=split_name)
        return metrics

    @dist_utils.main_process
    def _report_metrics(self, result_file, split):
        # 讀取結果
        with open(result_file, "r", encoding="utf-8") as f:
            records = json.load(f)

        # 分成可回答與不可能題
        answerable = [r for r in records if r["gt_list"]]
        impossible  = [r for r in records if not r["gt_list"]]

        # 可回答題指標
        hit1_count = 0
        exact_match_count = 0
        precisions = []
        recalls = []
        f1s = []

        for r in answerable:
            pred_set = set(r["pred_list"])
            gt_set   = set(r["gt_list"])
            # Hit@1
            if r["pred_list"] and r["pred_list"][0] in gt_set:
                hit1_count += 1
            # Exact Match
            if pred_set == gt_set:
                exact_match_count += 1
            # Precision
            precision = len(pred_set & gt_set) / len(pred_set) if r["pred_list"] else 0.0
            # Recall
            recall = len(pred_set & gt_set) / len(gt_set)
            # F1
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

        total_ans = len(answerable)
        metrics_ans = {
            "hit@1":       hit1_count / total_ans * 100 if total_ans else 0.0,
            "exact_match": exact_match_count / total_ans * 100 if total_ans else 0.0,
            "precision":   statistics.mean(precisions) * 100 if precisions else 0.0,
            "recall":      statistics.mean(recalls)    * 100 if recalls    else 0.0,
            "f1":          statistics.mean(f1s)        * 100 if f1s        else 0.0
        }

        # 不可能題指標
        impossible_correct = sum(1 for r in impossible if not r["pred_list"])
        total_imp = len(impossible)
        metrics_imp = {
            "impossible_accuracy": impossible_correct / total_imp * 100 if total_imp else 0.0
        }

                # 合併報表，並加入 agg_metrics (使用 f1 作為聚合指標)
        metrics = {
            **metrics_ans,
            **metrics_imp,
            "agg_metrics": metrics_ans["f1"]
        }

        # Logging
        log_path = os.path.join(registry.get_path("output_dir"), "log.txt")
        with open(log_path, "a") as f:
            f.write(json.dumps(metrics) + "\n")
        logging.info(f"[BadmintonQA] {split} metrics: {metrics}")
        return metrics


@registry.register_task("vizwiz")
class VizWizTask(VQATask):

    def build_datasets(self, cfg):
        datasets = super().build_datasets(cfg)

        # get question file, annotation file and anwser list in COCO format
        for dataset in datasets.values():
            for split in dataset:
                try:
                    self.answer_list = dataset[split].answer_list
                except AttributeError:
                    # if answer_list is not provided, then set it to None
                    pass

        if len(self.ques_files) > 0:
            assert len(self.ques_files) == len(
                self.anno_files
            ), "Only support one split for evaluation."

        return datasets
    
    def train_step(self, model, samples):
        txtout = []
        for i in range(len(samples["text_output"])):
            txtout.extend(samples["text_output"][i])
        sample_final = {"image" : torch.cat([samples["image"]] *10 ), "text_input": samples["text_input"]*10, "text_output": txtout}
    
        output = model(sample_final)
        loss_dict = {}
        for k,v in output.items():
            if "loss" in k:
                loss_dict[k] = v

        return output["loss"], loss_dict
    
    def valid_step(self, model, samples):
        answers = model.predict_answers(
            samples=samples,
            answer_list=self.answer_list,
            inference_method=self.inference_method,
            num_beams=self.num_beams,
            max_len=self.max_len,
            min_len=self.min_len,
            num_ans_candidates=self.num_ans_candidates,
            prompt=self.prompt,
        )
        pred_qa_pairs = []

        question_id = samples["question_id"]
        img_names = samples["image_name"]
        for answer, img_name in zip(answers, img_names):
            # ques_id = int(ques_id)
            pred_qa_pairs.append({"image": img_name, "answer": answer})

        return pred_qa_pairs

    def after_evaluation(self, val_result, split_name, **kwargs):
        result_file = self.save_result(
            val_result,
            result_dir=registry.get_path("result_dir"),
            filename=f"{split_name}_vqa_result",
            remove_duplicate="",
        )
        if split_name == 'val':
            metrics = self._report_metrics(result_file=result_file, split=split_name)
        else:
            metrics = None 
        return metrics

    @dist_utils.main_process
    def _report_metrics(self, result_file, split):
        """
        Use official Vizwiz evaluation script to report metrics.
        """
        metrics = {}
        print(result_file)
         
        annFile = "../../../input/disk-50gb/vizwiz/annotations/" + split + ".json"
        vqa = VQA_Vizwiz(annFile)
        vqa_result = VQA_Vizwiz(result_file)

        # create vqaEval object by taking vqa and vqaRes
        # n is precision of accuracy (number of places after decimal), default is 2
        vqa_scorer = VQAEval_Vizwiz(vqa, vqa_result, n=2)
        logging.info("Start VQA Vizwiz evaluation.")
        vqa_scorer.evaluate()

        # print accuracies
        overall_acc = vqa_scorer.accuracy["overall"]
        metrics["agg_metrics"] = overall_acc

        logging.info("Overall Accuracy is: %.02f\n" % overall_acc)
        logging.info("Per Answer Type Accuracy is the following:")

        for ans_type in vqa_scorer.accuracy["perAnswerType"]:
            logging.info(
                "%s : %.02f"
                % (ans_type, vqa_scorer.accuracy["perAnswerType"][ans_type])
            )
            metrics[ans_type] = vqa_scorer.accuracy["perAnswerType"][ans_type]

        with open(
            os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
        ) as f:
            print(f"wrote result on {f}")
            f.write(json.dumps(metrics) + "\n")

        return metrics

    
@registry.register_task("iconqa")
class IconQATask(VQATask):

    def build_datasets(self, cfg):
        datasets = super().build_datasets(cfg)

        # get question file, annotation file and anwser list in COCO format
        for dataset in datasets.values():
            for split in dataset:
                try:
                    self.answer_list = dataset[split].answer_list
                except AttributeError:
                    # if answer_list is not provided, then set it to None
                    pass

        if len(self.ques_files) > 0:
            assert len(self.ques_files) == len(
                self.anno_files
            ), "Only support one split for evaluation."

        return datasets

    # def train_step(self, model, samples):
    #     txtout = []
    #     for i in range(len(samples["text_output"])):
    #         txtout.extend(samples["text_output"][i])
    #     sample_final = {"image" : torch.cat([samples["image"]] *10 ), "text_input": samples["text_input"]*10, "text_output": txtout}
    
    #     output = model(sample_final)
    #     loss_dict = {}
    #     for k,v in output.items():
    #         if "loss" in k:
    #             loss_dict[k] = v

    #     return output["loss"], loss_dict
    
        # answers = model.predict_answers(
        #     samples=samples,
        #     answer_list=self.answer_list,
        #     inference_method=self.inference_method,
        #     num_beams=self.num_beams,
        #     max_len=self.max_len,
        #     min_len=self.min_len,
        #     num_ans_candidates=self.num_ans_candidates,
        #     prompt=self.prompt,
        # )
        # pred_qa_pairs = []

        # question_id = samples["question_id"]
        # img_names = samples["image_name"]
        # for answer, img_name in zip(answers, img_names):
        #     # ques_id = int(ques_id)
        #     pred_qa_pairs.append({"image": img_name, "answer": answer})

        # return pred_qa_pairs
    

    def valid_step(self, model, samples):
        # make predicted answers
        # answers = model.predict_answers(
        #     samples=samples,
        #     answer_list=self.answer_list,
        #     inference_method=self.inference_method,
        #     num_beams=self.num_beams,
        #     max_len=self.max_len,
        #     min_len=self.min_len,
        #     num_ans_candidates=self.num_ans_candidates,
        #     prompt=self.prompt,
        # )
        candidates = []
        if not isinstance(samples, list):
            i = 0
            for choice in samples["choices"][0]:
                label = chr(ord('a') + i)
                candidates.append(f"({label}) {choice}")
                i += 1
        else:
            candidates = samples['choices']
        
        answers = model.predict_class(
            samples=samples,
            candidates=candidates,
            n_segments=1,
        )
        pred_qa_pairs = []


        question_id = samples["question_id"]
        gt_answers = samples["answer"]
        # img_names = samples["image_name"]
        for pred_answer, ques_id, gt_answer in zip(answers, question_id, gt_answers):
            # ques_id = int(ques_id)
            pred_qa_pairs.append({"question_id": ques_id, "pred_ans": pred_answer, "gt_ans": gt_answer})

        return pred_qa_pairs

    def after_evaluation(self, val_result, split_name, **kwargs):
        # print(val_result[:5])
        result_file = self.save_result(
            val_result,
            result_dir=registry.get_path("result_dir"),
            filename=f"{split_name}_iconqa_result",
            remove_duplicate="",
        )
        if split_name == 'val':
            metrics = self._report_metrics(result_file=result_file, split=split_name)
        else:
            metrics = None 
        return metrics

    @dist_utils.main_process
    def _report_metrics(self, result_file, split):
        # IconeQA metric is easy
        # just check if the predicted answer is the ground truth answer

        results = json.load(open(result_file, "r"))
        acc = []

        for res in results:
            # if res["gt_ans"] is None:
            #     # prepare test results for leaderboard evaluation
            #     self._save_result_leaderboard(results)
            #     return

            pred = res["pred_ans"]
            gt_ans = [res["gt_ans"]]

            num_match = sum([pred == gt for gt in gt_ans])
            vqa_acc = min(1.0, num_match / len(gt_ans))

            acc.append(vqa_acc)

        accuracy = sum(acc) / len(acc) * 100
        metrics = {"agg_metrics": accuracy, "acc": accuracy}

        with open(
            os.path.join(registry.get_path("output_dir"), f"log.txt"), "a"
        ) as f:
            f.write(json.dumps(metrics) + "\n")

        logging.info(metrics)

        return metrics