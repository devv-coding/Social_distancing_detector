# python SDD_Video+Webcam.py --input "rtsp://admin:WorldSpa@12345@worldspa.ddns.net:3556/Streaming/Channels/101?transportmode=unicast&profile=Profile_1"
# python SDD_Video+Webcam.py --input output_test2.avi --output output4.mp4
# python SDD_Video+Webcam.py --input output_test3_Trim.mp4
import social_distance_cfg as config
from detect import detect_people
from scipy.spatial import distance as dist
import numpy as np
import argparse
import imutils
import cv2
# import cv
import os
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
x = 0
y = 0
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default="",
	help="path to (optional) input video file")
ap.add_argument("-o", "--output", type=str, default="",
	help="path to (optional) output video file")
ap.add_argument("-d", "--display", type=int, default=1,
	help="whether or not output frame should be displayed")
args = vars(ap.parse_args())

labelsPath = os.path.sep.join([config.MODEL_PATH, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

weightsPath = os.path.sep.join([config.MODEL_PATH, "yolov3.weights"])
configPath = os.path.sep.join([config.MODEL_PATH, "yolov3.cfg"])

print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

if config.USE_GPU:
	print("[INFO] setting preferable backend and target to CUDA...")
	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
	net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

print("[INFO] accessing video stream...")
vs = cv2.VideoCapture(args["input"] if args["input"] else 0, cv2.CAP_FFMPEG)
# vs.set(cv2.CAP_PROP_BUFFERSIZE, 3)
# # cap.set(cv::CAP_PROP_BUFFERSIZE, 3)
# vs.set(cv2.CAP_PROP_FPS, 9)
violate = set()
common = set()

fps = vs.get(cv2.CAP_PROP_FPS)
writer = None

while True:
	(grabbed, frame) = vs.read()

	if not grabbed: 
		vs = cv2.VideoCapture(args["input"] if args["input"] else 0, cv2.CAP_FFMPEG)  
		continue

	frame = imutils.resize(frame, width=1100, height=1400)
	x=x+1
	results = detect_people(frame, net, ln,
		personIdx=LABELS.index("person"))

	# violate = set()
	# common = set()
	violate_current = set()

	if len(results) >= 2:
		centroids = np.array([r[2] for r in results])
		D = dist.cdist(centroids, centroids, metric="euclidean")
		# cv2.line(frame, tuple(centroids), tuple(centroids), color=(0, 0, 0), thickness=3)

		for i in range(0, D.shape[0]):
			for j in range(i + 1, D.shape[1]):
				if D[i, j] < config.MIN_DISTANCE:	
					violate_current.add(i)# violate.add(j)
					violate_current.add(j)# violate.add(j)														
					common.add(i)
					if len(violate.intersection(common)) == 0:
						violate.add(i)
					common.clear()					
					common.add(j)
					if len(violate.intersection(common)) == 0:
						violate.add(j)
					common.clear()						
					cv2.line(frame, tuple(centroids[i]), tuple(centroids[j]), color=(51, 51, 255), thickness=2)

	for (i, (prob, bbox, centroid)) in enumerate(results):
		(startX, startY, endX, endY) = bbox
		(cX, cY) = centroid
		color = (0, 255, 0)

		if i in violate_current: color = (0, 0, 255); text = "Coming Close"; cv2.putText(frame, text, (startX, startY - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 255), 1); cv2.circle(frame, (cX, cY), 5, color, 1) 
		# cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
		# cv2.circle(frame, (cX, cY), 5, color, 2)

	print("Frame ID : ",x)
	y = len(violate) # + y
	print("Social Distancing Violations: ",y)
	text = "Social Distancing Violations: {} & FPS: {}".format(y,fps)
	cv2.putText(frame, text, (10, frame.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3), 

	if args["display"] > 0: 
		cv2.imshow("Social_Distancing_Detector_AnjaliDubey", frame) 
		key = cv2.waitKey(1) & 0xFF; 
		if key == ord("q"): break

	if args["output"] != "" and writer is None:
		fourcc = cv2.VideoWriter_fourcc(*"FMP4") #MP4V
		writer = cv2.VideoWriter(args["output"], fourcc, 25,
			(frame.shape[1], frame.shape[0]), True)

	if writer is not None:
		writer.write(frame)

#writer.release() frame.shape[1], frame.shape[0]
vs.release()
cv2.destroyAllWindows()