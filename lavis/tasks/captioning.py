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

        agg_metrics = badminton_eval.eval["CIDEr"] + badminton_eval.eval["SPICE"]
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

    badminton = COCO(annotation_file)
    badminton_result = badminton.loadRes(results_file)

    # create COCOEvalCap object by taking badminton and badminton_result
    badminton_eval = COCOEvalCap(badminton, badminton_result)

    # evaluate on a subset of images by setting
    # badminton_eval.params["image_id"] = badminton_result.getImgIds()  
    # ↑ 如果要評估整個驗證集，請移除此行

    # evaluate results
    badminton_eval.evaluate()

    # print CIDEr and SPICE scores
    print(f"CIDEr: {badminton_eval.eval['CIDEr']:.3f}")
    print(f"SPICE: {badminton_eval.eval['SPICE']:.3f}")

    return badminton_eval
