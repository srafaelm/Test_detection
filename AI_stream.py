from sensor_streaming import ar_player
import copy
import cv2
import time
import numpy as np
import json

import math

import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

import skimage
from sort import *

class AIServer:
  def __init__(self, HoloLensIP):
    self.ActiveClassList = ""
    self.ObjectList = []

    self.host = HoloLensIP
    #self.resolution = resolution
    self.sensor_stream = None

    # Load AI modules
    set_logging()
    self.device = select_device('')  # Use the default CUDA device if available
    half = self.device.type != 'cpu'  # half precision only supported on CUDA

    self.conf_threshold = 0.5

    sort_max_age = 5
    sort_min_hits = 2
    sort_iou_thresh = 0.25
    self.sort_tracker = Sort(max_age=sort_max_age,
                        min_hits=sort_min_hits,
                        iou_threshold=sort_iou_thresh)

    weights_path = '/home/goncalo/rafa_docs/test_detection/weights/yolov7.pt'  # Adjust to your model path

    self.model = attempt_load(weights_path, map_location=self.device)
    stride = int(self.model.stride.max())  # model stride
    imgsz = check_img_size(640, s=stride)  # check img_size

    if half:
      self.model.half()  # to FP16

    if self.device.type != 'cpu':
      self.model(torch.zeros(1, 3, 640, 640).to(self.device).type_as(next(self.model.parameters())))  # run once
      old_img_w = old_img_h = imgsz
      old_img_b = 1
      print('Here mudou')

    # Load calibration data (intrinsics and extrinsics) --> fixed_rgb has reversed directions for x and y camera coordinates
    #self.depth_calibration, self.rgb_calibration, self.fixed_rgb_calibration = ar_player.load_calibration_data(
      #"calibration_data/", self.resolution)
    self.depth_calibration, self.rgb_calibration, self.fixed_rgb_calibration = ar_player.load_calibration_data(
      "calibration_data/")

  def ServerStart(self):
    if self.sensor_stream is not None:
      return "Error. Server already started"

    # Start sensor streams
    self.sensor_stream = ar_player.SensorStreamer(self.host)

    # Wait for the sensor stream to begin
    print("Waiting for the data streams")
    while self.sensor_stream.rgb is None:
      continue

    return "Server started"

  def ServerSetActiveClassList(self, new_classes):
    self.ActiveClassList = new_classes
    #print("ActiveClassList is", new_classes)
    return "ActiveClassList set"

  def ServerGetObjectList(self, requested_classes):
    if len(requested_classes) == 0:
      return "Error. Active Class List is empty"

    self.ObjectList = []

    # Access latest rgb and depth images (copy to avoid overwrite)
    rgb = copy.deepcopy(self.sensor_stream.rgb)  # rgb.payload is the image
    depth = copy.deepcopy(self.sensor_stream.depth.payload.depth)
    #print('chegou')
    # Perform OCR on measuring instruments
    if "ocr.measuring" in requested_classes:
      # Perform OCR on the rgb image
      original=rgb.payload
      samer = copy.deepcopy(rgb.payload)
      dim = (640, 640)
      img = cv2.resize(original, dim, interpolation=cv2.INTER_AREA)

      img = torch.from_numpy(img).to(self.device)
      original = torch.from_numpy(original).to(self.device)
      original = original.permute(2, 0, 1).unsqueeze(0)
      half = self.device.type != 'cpu'

      img = img.half() if half else img.float()  # uint8 to fp16/32
      img /= 255.0  # 0 - 255 to 0.0 - 1.0
      img = img.permute(2, 0, 1).unsqueeze(0)  # Change the layout to [1, 3, 640, 640]
      #print(img.shape)
      if img.ndimension() == 3:
        img = img.unsqueeze(0)
        #print(img.shape, 'img after')
      with torch.no_grad():

        results = self.model(img, augment=True)[0]
      results = non_max_suppression(results, self.conf_threshold, 0.45)
      undistorted_depth, u, v = ar_player.get_depth_to_pv_map(self.depth_calibration, self.rgb_calibration, depth,rgb.payload)
      for det in enumerate(results):
        if len(det):

          # print(scale_coords(img.shape[2:], det[:, :4],  frame.shape).round(), 'shuflle')
          det = det[1]
          # print(det[:1, 1], 'det personalized')
          # print(det[:1, 1], 'det personalized')
          #print(img.shape, 'image shape')
          #print(original.shape, 'orignal shape')
          det[:, :4] = scale_coords(img.shape[2:], det[:, :4], original.shape[2:]).round()
          time_of_bbox = time.time()


          validated_texts = []
          validated_pixel_boxes = []
          confidance=[]
          for *xyxy, conf, cls in det:
            label = f'{self.model.names[int(cls)]} {conf:.2f}'
            plot_one_box(xyxy, samer, label=label, color=(0, 255, 0))  # You can customize the box color
            confidance.append(conf)
            #print(xyxy, 'xyxy')
            #x1=int(xyxy[0].item())
            #y1=int(xyxy[1].item())
            #x2=int(xyxy[2].item())
            #y2=int(xyxy[3].item())


            ###### TRACKER ######
          dets_to_sort = np.empty((0, 6))
          for x1_t, y1_t, x2_t, y2_t, conf_t, detclass_t in det.cpu().detach().numpy():
            dets_to_sort = np.vstack((dets_to_sort,
                                      np.array([x1_t, y1_t, x2_t, y2_t, conf_t, detclass_t])))

          self.tracked_dets = self.sort_tracker.update(dets_to_sort)
          print(self.tracked_dets, 'dets to sort')
          tracks = self.sort_tracker.getTrackers()
          if len(self.tracked_dets) > 0:
            #print(self.tracked_dets[:, :4], 'bbox###############################################################################################')
            #print(self.tracked_dets[:, 8], 'idenities')
            print(self.tracked_dets, 'categories')
            counter = 0
            for jj, box_tr in enumerate(self.tracked_dets[:, :5]):
              counter +=1
              if any(math.isnan(coord) for coord in box_tr):
                # Skip this iteration if any coordinate is NaN
                continue
              x1, y1, x2, y2, classe = [int(jj) for jj in box_tr]
              print(box_tr)
              label = f'{self.model.names[classe]} {confidance[counter-1]:.2f}'
              bbox = np.array([[x1, y2], [x2, y2], [x2, y1], [x1, y1]])
              bbox_3d = ar_player.get_3d_bbox(bbox, undistorted_depth, u, v, self.fixed_rgb_calibration)
              if bbox_3d is None:
                continue

              validated_texts.append(label)
              validated_pixel_boxes.append(bbox)
              world_bbox = ar_player.transform_to_world(bbox_3d, rgb.pose)
              self.ObjectList.append({"classID": "measuring_data",
                                      "world_box": world_bbox.tolist(),
                                      "text": label,
                                      "camera_ref_box": bbox_3d.tolist(),
                                      "head_pose": rgb.pose.tolist(),
                                      "pixel_box": bbox.tolist(),
                                      "rgb_timestamp": rgb.timestamp,
                                      "unix_timestamp": time_of_bbox
                                      })
              #print(bbox, 'bbox')

              #bbox_3d= np.array([[x1, y2, 1], [x2, y2, 1], [x2, y1, 1], [x1, y1, 1]])

              # Bounding boxes of points without a depth correspondence will be ignored



              #print(results)
              #print(label)
              # Transform bbox to 3D world space coordinates -> bbox is shaped (5, 3)
              # the first dimension corresponds to each point in order topleft, topright, bottomright, bottomleft, center
              # the second dimension are the coordinates of each point in order x y z

              #world_bbox=bbox_3d
              # Add obtained bbox and text to ObjectList


        else:  # SORT should be updated even with no detections
          self.tracked_dets = self.sort_tracker.update()

          cv2.imwrite('/home/goncalo/rafa_docs/hl2ss-main/testing/' + str(rgb.timestamp) + '.jpeg', samer)
      ################################################################




      new_frame = copy.deepcopy(rgb.payload)
      # cv2.putText(new_frame, "Head pose: " + str(rgb.pose), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
      # cv2.putText(new_frame, "RGB_timestamp: " + str(rgb.timestamp), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
      #             (255, 0, 0), 1)
      # y = 350
      # dy = 30
      # for j, box in enumerate(validated_pixel_boxes):
      #   cv2.putText(new_frame, "Pixel: " + str(box) + " --> " + validated_texts[j], (10, y), cv2.FONT_HERSHEY_SIMPLEX,
      #               0.4, (255, 0, 0), 1)
      #   y = y - dy
      #   cv2.putText(new_frame, validated_texts[j], (int(box[1][0]), int(box[1][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
      #               (0, 0, 255), 1)
      # cv2.polylines(new_frame, np.int32(validated_pixel_boxes), True, (0, 0, 255), 1)
      # cv2.imwrite("OCR_results/" + str(int(time_of_bbox)) + ".jpg", new_frame)

    # Change ObjectList to desired JSON syntax
    self.ObjectList = json.dumps(self.ObjectList, indent=2, sort_keys=True)
    return self.ObjectList

  def ServerStop(self):
    if self.sensor_stream is None:
      return "Error. Server not started"

    # Close sensor streams
    self.sensor_stream.enable_streams = False
    del self.sensor_stream  # delete instance of streamer
    self.sensor_stream = None
    return "Server stopped"



