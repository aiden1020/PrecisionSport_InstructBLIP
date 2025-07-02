"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import json
import os
import tempfile

from lavis.common.dist_utils import main_process
from lavis.common.registry import registry
from lavis.tasks.base_task import BaseTask
import re

@registry.register_task("captioning")
class CaptionTask(BaseTask):
    def __init__(self, num_beams, max_len, min_len, evaluate, report_metric=True):
        super().__init__()

        self.num_beams = num_beams
        self.max_len = max_len
        self.min_len = min_len
        self.evaluate = evaluate

        self.report_metric = report_metric

    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg

        num_beams = run_cfg.num_beams
        max_len = run_cfg.max_len
        min_len = run_cfg.min_len
        evaluate = run_cfg.evaluate

        report_metric = run_cfg.get("report_metric", True)

        return cls(
            num_beams=num_beams,
            max_len=max_len,
            min_len=min_len,
            evaluate=evaluate,
            report_metric=report_metric,
        )

    def valid_step(self, model, samples):
        results = []

        # run_cfg = slf.cfg.run_cfg
        captions = model.generate(
            samples,
            use_nucleus_sampling=False,
            num_beams=self.num_beams,
            max_length=self.max_len,
            min_length=self.min_len,
        )

        img_ids = samples["image_id"]
        for caption, img_id in zip(captions, img_ids):
            results.append({"caption": caption, "image_id": int(img_id)})

        return results

    def after_evaluation(self, val_result, split_name, epoch, **kwargs):
        eval_result_file = self.save_result(
            result=val_result,
            result_dir=registry.get_path("result_dir"),
            filename="{}_epoch{}".format(split_name, epoch),
            remove_duplicate="image_id",
        )

        if self.report_metric:
            metrics = self._report_metrics(
                eval_result_file=eval_result_file, split_name=split_name
            )
        else:
            metrics = {"agg_metrics": 0.0}

        return metrics

    @main_process
    def _report_metrics(self, eval_result_file, split_name):
        # TODO better way to define this
        coco_gt_root = os.path.join(registry.get_path("cache_root"), "coco_gt")
        coco_val = coco_caption_eval(coco_gt_root, eval_result_file, split_name)

        agg_metrics = coco_val.eval["CIDEr"]
        log_stats = {split_name: {k: v for k, v in coco_val.eval.items()}}

        with open(
            os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
        ) as f:
            f.write(json.dumps(log_stats) + "\n")

        coco_res = {k: v for k, v in coco_val.eval.items()}
        coco_res["agg_metrics"] = agg_metrics

        return coco_res


@registry.register_task("flickr30k_instruct")
class Flickr30kCaptionTask(CaptionTask):
    def valid_step(self, model, samples):
        results = []

        captions = model.generate(
            samples,
            use_nucleus_sampling=False,
            num_beams=self.num_beams,
            max_length=self.max_len,
            min_length=self.min_len,
        )

        img_ids = samples["image_id"]
        for caption, img_id in zip(captions, img_ids):
            results.append({"caption": caption, "image_id": int(img_id)})

        return results

    def after_evaluation(self, val_result, split_name, epoch, **kwargs):
        eval_result_file = self.save_result(
            val_result,
            result_dir=registry.get_path("result_dir"),
            filename=f"{split_name}_flickr30k_caption_instruct_result_epoch{epoch}",
            remove_duplicate="",
        )
        if split_name == "val":
            metrics = self._report_metrics(
                eval_result_file=eval_result_file, split_name=split_name
            )
        else:
            metrics = None
        return metrics

    @main_process
    def _report_metrics(self, eval_result_file, split_name):
        coco_val = flickr30k_caption_eval(eval_result_file, split_name)

        agg_metrics = coco_val.eval["CIDEr"] + coco_val.eval["Bleu_4"]
        log_stats = {split_name: {k: v for k, v in coco_val.eval.items()}}

        with open(
            os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
        ) as f:
            f.write(json.dumps(log_stats) + "\n")

        coco_res = {k: v for k, v in coco_val.eval.items()}
        coco_res["agg_metrics"] = agg_metrics

        return coco_res

@registry.register_task("badminton_caption")
class BadmintonCaptionTask(CaptionTask):
    def valid_step(self, model, samples):
        results = []
        captions = model.generate(
            samples,
            use_nucleus_sampling=False,
            num_beams=self.num_beams,
            max_length=self.max_len,
            min_length=self.min_len,
        )

        img_ids = samples["image_id"]
        for caption, img_id in zip(captions, img_ids):
            results.append({"caption": caption, "image_id": img_id})
        return results

    def after_evaluation(self, val_result, split_name, epoch, **kwargs):
        eval_result_file = self.save_result(
            val_result,
            result_dir=registry.get_path("result_dir"),
            filename=f"{split_name}_badminton_caption_result_epoch{epoch}",
            remove_duplicate="",
        )
        if split_name == "val":
            metrics = self._report_metrics(
                eval_result_file=eval_result_file, split_name=split_name
            )
        else:
            metrics = None
        # metrics = None

        return metrics

    @main_process
    def _report_metrics(self, eval_result_file, split_name):
        badminton_eval = badminton_caption_eval(eval_result_file, split_name)

        agg_metrics = (badminton_eval.stroke_acc + badminton_eval.area_acc)/2
        log_stats = {split_name: {k: v for k, v in badminton_eval.eval.items()}}

        with open(
            os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
        ) as f:
            f.write(json.dumps(log_stats) + "\n")

        badminton_res = {k: v for k, v in badminton_eval.eval.items()}
        badminton_res["agg_metrics"] = agg_metrics

        return badminton_res

# TODO better structure for this.
from pycocoevalcap.eval import COCOEvalCap
from pycocotools.coco import COCO
from torchvision.datasets.utils import download_url


def coco_caption_eval(coco_gt_root, results_file, split):
    urls = {
        "val": "https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val_gt.json",
        "test": "https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test_gt.json",
    }
    filenames = {
        "val": "coco_karpathy_val_gt.json",
        "test": "coco_karpathy_test_gt.json",
    }

    download_url(urls[split], coco_gt_root)
    annotation_file = os.path.join(coco_gt_root, filenames[split])

    # create coco object and coco_result object
    coco = COCO(annotation_file)
    coco_result = coco.loadRes(results_file)

    # create coco_eval object by taking coco and coco_result
    coco_eval = COCOEvalCap(coco, coco_result)

    # evaluate on a subset of images by setting
    # coco_eval.params['image_id'] = coco_result.getImgIds()
    # please remove this line when evaluating the full validation set
    # coco_eval.params['image_id'] = coco_result.getImgIds()

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    coco_eval.evaluate()

    # print output evaluation scores
    for metric, score in coco_eval.eval.items():
        print(f"{metric}: {score:.3f}")

    return coco_eval


def flickr30k_caption_eval(results_file, split):
    files = {
        "val": "/input/flickr30k/annotations/val_gt.json",
        "test": "/input/flickr30k/annotations/test_gt.json",
    }
    annotation_file = files[split]

    flickr = COCO(annotation_file)
    print(f"flickr: {annotation_file}")
    print(f"results: {results_file}")
    flickr_result = flickr.loadRes(results_file)

    # create coco_eval object by taking flickr and flickr_result
    flickr_eval = COCOEvalCap(flickr, flickr_result)

    # evaluate on a subset of images by setting
    flickr_eval.params[
        "image_id"
    ] = (
        flickr_result.getImgIds()
    )  # please remove this line when evaluating the full validation set

    # evaluate results
    flickr_eval.evaluate()

    # print CIDEr output evaluation scores
    print(f"CIDEr: {flickr_eval.eval['CIDEr']:.3f}")

    return flickr_eval

def badminton_caption_eval(results_file, split):
    files = {
        "val": "lavis/configs/datasets/badminton_caption/input/val_gt.json",
        "test": "lavis/configs/datasets/badminton_caption/input/test_gt.json",
    }
    annotation_file = files[split]
    coco_gt  = COCO(annotation_file)
    coco_res = coco_gt.loadRes(results_file)

    coco_eval = COCOEvalCap(coco_gt, coco_res)
    coco_eval.evaluate()
    print(f"CIDEr: {coco_eval.eval['CIDEr']:.3f}")
    print(f"SPICE: {coco_eval.eval['SPICE']:.3f}")

    pattern = r'(.+?) hits a (.+?) (?:at|in|on) (?:the )?(.+?)(?:\.|$)'
    def extract_parts(caption):
        m = re.match(pattern, caption.lower())
        if not m:
            return None
        _, stroke, area = m.groups()
        stroke = stroke.replace('-', ' ').strip()
        area   = area.replace('-', ' ').strip()
        return stroke, area

    img_ids = coco_gt.getImgIds()
    total = len(img_ids)
    stroke_correct = 0
    area_correct   = 0

    for img_id in img_ids:
        res_ann_ids = coco_res.getAnnIds(imgIds=[img_id])
        res_anns    = coco_res.loadAnns(res_ann_ids)
        if not res_anns:
            continue
        pred = extract_parts(res_anns[0]['caption'])
        if not pred:
            continue
        pred_stroke, pred_area = pred

        gt_ann_ids = coco_gt.getAnnIds(imgIds=[img_id])
        gt_anns    = coco_gt.loadAnns(gt_ann_ids)
        ref_strokes = set()
        ref_areas   = set()
        for ann in gt_anns:
            parts = extract_parts(ann['caption'])
            if parts:
                s, a = parts
                ref_strokes.add(s)
                ref_areas.add(a)

        if pred_stroke in ref_strokes:
            stroke_correct += 1
        if pred_area in ref_areas:
            area_correct += 1

    stroke_acc = stroke_correct / total if total else 0
    area_acc   = area_correct   / total if total else 0
    coco_eval.stroke_acc = stroke_acc
    coco_eval.area_acc   = area_acc
    print("Stroke accuracy:", coco_eval.stroke_acc)
    print("Area accuracy:  ", coco_eval.area_acc)
    return coco_eval
