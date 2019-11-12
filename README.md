# C++ Yolo v3 Object Detection
	Detect objects from image, can be used for videos and streams
	
# Prerequisites
	OpenCV installed
	Download yolo v3 tiny files from the internet and add to "files" directory:
		- yolov3-tiny.weights
		- yolov3-tiny.cfg
		- coco.names
		
# Usage - Blits image from your desktop and detects objects
	- Yolo yolo;
	- yolo.setup(416, 416);
	- BitBlt(yolo.hdcMem, 0, 0, yolo.x_size, yolo.y_size, yolo.hdcSys, (int)(screenW / 2) - (yolo.x_size / 2), (int)(screenH / 2) - (yolo.y_size / 2), SRCCOPY);
	- yolo.detect(yolo.matBitmap);
	- cout << "DETECTIONS: " << yolo.faces.size() << endl;