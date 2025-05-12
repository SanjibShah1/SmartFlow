# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

import argparse
import cv2
import numpy as np
from functools import partial
from pathlib import Path
from collections import defaultdict

import torch

from boxmot import TRACKERS
from boxmot.tracker_zoo import create_tracker
from boxmot.utils import ROOT, WEIGHTS, TRACKER_CONFIGS
from boxmot.utils.checks import RequirementsChecker
from tracking.detectors import (get_yolo_inferer, default_imgsz,
                                is_ultralytics_model, is_yolox_model)

checker = RequirementsChecker()
checker.check_packages(('ultralytics @ git+https://github.com/mikel-brostrom/ultralytics.git', ))  # install

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from ultralytics.data.utils import VID_FORMATS
from ultralytics.utils.plotting import save_one_box

# Global variables for lane and direction counting
vehicle_tracker = defaultdict(lambda: {
    'positions': [],
    'counted': False,
    'lane': None,
    'direction': None
})
lane_outgoing = {'two_wheeler': 0, 'four_wheeler': 0}  # Right lane (Out)
lane_incoming = {'two_wheeler': 0, 'four_wheeler': 0}  # Left lane (In)
counting_line_y = 350  # Adjust based on your video
num_lanes = 2  # Left and Right lanes


def on_predict_start(predictor, persist=False):
    """
    Initialize trackers for object tracking during prediction.

    Args:
        predictor (object): The predictor object to initialize trackers for.
        persist (bool, optional): Whether to persist the trackers if they already exist. Defaults to False.
    """

    assert predictor.custom_args.tracking_method in TRACKERS, \
        f"'{predictor.custom_args.tracking_method}' is not supported. Supported ones are {TRACKERS}"

    tracking_config = TRACKER_CONFIGS / (predictor.custom_args.tracking_method + '.yaml')
    trackers = []
    for i in range(predictor.dataset.bs):
        tracker = create_tracker(
            predictor.custom_args.tracking_method,
            tracking_config,
            predictor.custom_args.reid_model,
            predictor.device,
            predictor.custom_args.half,
            predictor.custom_args.per_class
        )
        # motion only modeles do not have
        if hasattr(tracker, 'model'):
            tracker.model.warmup()
        trackers.append(tracker)

    predictor.trackers = trackers


@torch.no_grad()
def run(args):
    global counting_line_y, num_lanes

    if args.imgsz is None:
        args.imgsz = default_imgsz(args.yolo_model)

    yolo = YOLO(
        args.yolo_model if is_ultralytics_model(args.yolo_model)
        else 'yolov8n.pt',
    )

    results = yolo.track(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        agnostic_nms=args.agnostic_nms,
        show=False,
        stream=True,
        device=args.device,
        show_conf=False,  # Hide confidence
        show_labels=False,  # Hide class labels
        save_txt=args.save_txt,
        save=args.save,
        verbose=args.verbose,
        exist_ok=args.exist_ok,
        project=args.project,
        name=args.name,
        classes=args.classes,
        imgsz=args.imgsz,
        vid_stride=args.vid_stride,
        line_width=args.line_width
    )

    yolo.add_callback('on_predict_start', partial(on_predict_start, persist=True))

    if not is_ultralytics_model(args.yolo_model):
        # replace yolov8 model
        m = get_yolo_inferer(args.yolo_model)
        yolo_model = m(model=args.yolo_model, device=yolo.predictor.device,
                       args=yolo.predictor.args)
        yolo.predictor.model = yolo_model

        # If current model is YOLOX, change the preprocess and postprocess
        if is_yolox_model(args.yolo_model):
            # add callback to save image paths for further processing
            yolo.add_callback("on_predict_batch_start",
                              lambda p: yolo_model.update_im_paths(p))
            yolo.predictor.preprocess = (
                lambda imgs: yolo_model.preprocess(im=imgs))
            yolo.predictor.postprocess = (
                lambda preds, im, im0s:
                yolo_model.postprocess(preds=preds, im=im, im0s=im0s))

    # store custom args in predictor
    yolo.predictor.custom_args = args

    for r in results:
        img = yolo.predictor.trackers[0].plot_results(r.orig_img, args.show_trajectories)

        if args.show or args.save:
            # Draw counting line
            cv2.line(img, (0, counting_line_y), (img.shape[1], counting_line_y), (0, 0, 255), 2)

            # Process each detected object
            for box in r.boxes:
                if box.id is None:
                    continue

                track_id = int(box.id.item())
                xyxy = box.xyxy[0].cpu().numpy()
                x_center = (xyxy[0] + xyxy[2]) / 2
                y_center = (xyxy[1] + xyxy[3]) / 2
                cls = int(box.cls.item())  # Class ID (0: Four Wheeler, 1: Two Wheeler)

                # Update position history
                vehicle = vehicle_tracker[track_id]
                vehicle['positions'].append((x_center, y_center))
                if len(vehicle['positions']) > 5:
                    vehicle['positions'].pop(0)

                # Determine lane (left or right)
                lane = "left" if x_center < img.shape[1] / 2 else "right"
                vehicle['lane'] = lane

                # Determine direction
                if len(vehicle['positions']) >= 2:
                    prev_y = vehicle['positions'][-2][1]
                    direction = "outgoing" if y_center < prev_y else "incoming"
                    vehicle['direction'] = direction

                    # Check counting line crossing
                    if (prev_y > counting_line_y and y_center <= counting_line_y) or \
                       (prev_y < counting_line_y and y_center >= counting_line_y):
                        if not vehicle['counted']:
                            if lane == "right":  # Outgoing (right lane)
                                if cls == 0:  # Four Wheeler
                                    lane_outgoing['four_wheeler'] += 1
                                elif cls == 1:  # Two Wheeler
                                    lane_outgoing['two_wheeler'] += 1
                            else:  # Incoming (left lane)
                                if cls == 0:  # Four Wheeler
                                    lane_incoming['four_wheeler'] += 1
                                elif cls == 1:  # Two Wheeler
                                    lane_incoming['two_wheeler'] += 1
                            vehicle['counted'] = True

            # Draw counts on frame
            # Incoming (Left Lane) - Top Left
            incoming_text = [
                f'In',
                f'Two Wheeler: {lane_incoming["two_wheeler"]}',
                f'Four Wheeler: {lane_incoming["four_wheeler"]}'
            ]
            for i, line in enumerate(incoming_text):
                cv2.putText(img, line, (20, 50 + i * 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Outgoing (Right Lane) - Top Right
            outgoing_text = [
                f'Out',
                f'Two Wheeler: {lane_outgoing["two_wheeler"]}',
                f'Four Wheeler: {lane_outgoing["four_wheeler"]}'
            ]
            for i, line in enumerate(outgoing_text):
                cv2.putText(img, line, (img.shape[1] - 200, 50 + i * 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if args.show:
            cv2.imshow('BoxMOT', img)
            if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
                break


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-model', type=Path, default=WEIGHTS / 'yolov8n',
                        help='yolo model path')
    parser.add_argument('--reid-model', type=Path, default=WEIGHTS / 'osnet_x0_25_msmt17.pt',
                        help='reid model path')
    parser.add_argument('--tracking-method', type=str, default='deepocsort',
                        help='deepocsort, botsort, strongsort, ocsort, bytetrack, imprassoc')
    parser.add_argument('--source', type=str, default='0',
                        help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=None,
                        help='inference size h,w')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='confidence threshold')
    parser.add_argument('--iou', type=float, default=0.7,
                        help='intersection over union (IoU) threshold for NMS')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show', action='store_true',
                        help='display tracking video results')
    parser.add_argument('--save', action='store_true',
                        help='save video tracking results')
    parser.add_argument('--classes', nargs='+', type=int,
                        help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--project', default=ROOT / 'runs' / 'track',
                        help='save results to project/name')
    parser.add_argument('--name', default='exp',
                        help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true',
                        help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true',
                        help='use FP16 half-precision inference')
    parser.add_argument('--vid-stride', type=int, default=1,
                        help='video frame-rate stride')
    parser.add_argument('--show-labels', action='store_false',
                        help='either show all or only bboxes')
    parser.add_argument('--show-conf', action='store_false',
                        help='hide confidences when show')
    parser.add_argument('--show-trajectories', action='store_true',
                        help='show confidences')
    parser.add_argument('--save-txt', action='store_true',
                        help='save tracking results in a txt file')
    parser.add_argument('--save-id-crops', action='store_true',
                        help='save each crop to its respective id folder')
    parser.add_argument('--line-width', default=None, type=int,
                        help='The line width of the bounding boxes. If None, it is scaled to the image size.')
    parser.add_argument('--per-class', default=False, action='store_true',
                        help='not mix up classes when tracking')
    parser.add_argument('--verbose', default=True, action='store_true',
                        help='print results per frame')
    parser.add_argument('--agnostic-nms', default=False, action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--num-lanes', type=int, default=2,
                        help='number of traffic lanes')
    parser.add_argument('--counting-line-y', type=int, default=350,
                        help='y-coordinate for counting line')

    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    counting_line_y = opt.counting_line_y
    num_lanes = opt.num_lanes
    run(opt)