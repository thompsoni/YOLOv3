#include "yolo.h"

Yolo::Yolo() {
}

void Yolo::setup(int width, int height) {

	// Load names of classes
	ifstream ifs(classesFile.c_str());
	string line;
	while (getline(ifs, line)) classes.push_back(line);

	// Load the network
	net = readNetFromDarknet(modelConfiguration, modelWeights);
	net.setPreferableBackend(DNN_BACKEND_OPENCV);
	net.setPreferableTarget(DNN_TARGET_CPU);

	// stream images from desktop
	////////////////////////////////////
	x_size = width;
	y_size = height;

	// Initialize DCs
	hdcSys = GetDC(NULL); // Get DC of the target capture..
	hdcMem = CreateCompatibleDC(hdcSys); // Create compatible DC 

	// Create hBitmap with Pointer to the pixels of the Bitmap
	ZeroMemory(&bi, sizeof(BITMAPINFO));
	bi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
	bi.bmiHeader.biWidth = x_size;
	bi.bmiHeader.biHeight = -y_size;  //negative so (0,0) is at top left
	bi.bmiHeader.biPlanes = 1;
	bi.bmiHeader.biBitCount = 32;
	hdc = GetDC(NULL);
	hBitmap = CreateDIBSection(hdc, &bi, DIB_RGB_COLORS, &ptrBitmapPixels, NULL, 0);
	// ^^ The output: hBitmap & ptrBitmapPixels

	// Set hBitmap in the hdcMem 
	SelectObject(hdcMem, hBitmap);

	// Set matBitmap to point to the pixels of the hBitmap
	matBitmap = Mat(y_size, x_size, CV_8UC4, ptrBitmapPixels, 0);
}

void Yolo::detect(Mat& img) {
	Mat blob;

	Mat gray, smallImg;
	cvtColor(img, gray, COLOR_BGR2GRAY);
	double fx = 1 / 1.0;
	resize(gray, smallImg, Size(), fx, fx, INTER_LINEAR_EXACT);
	equalizeHist(smallImg, smallImg);

	Mat im_coloured = Mat::zeros(img.rows, img.cols, CV_8UC3);
	vector<Mat> planes;
	for (int i = 0; i < 3; i++) {
		planes.push_back(smallImg);
	}
		//cout << "im_coloured" << x_size << "x" << y_size << endl;

	merge(planes, im_coloured);

	// Create a 4D blob from a frame.
	blobFromImage(im_coloured, blob, 1 / 255.0, cvSize(x_size, y_size), Scalar(0, 0, 0), true, false);
		//cout << "blob" << blob.size() << endl;

	//Sets the input to the network
	net.setInput(blob);
		//cout << "set input" << endl;

	// Runs the forward pass to get output of the output layers
	vector<Mat> outs;
	net.forward(outs, getOutputsNames(net));
		//cout << "outs" << outs.size() << endl;

	faces = getBoxes(img, outs);
	//imshow("",img); //debug screen
}

// Remove the bounding boxes with low confidence using non-maxima suppression
void Yolo::postprocess(Mat& frame, const vector<Mat>& outs)
{
	vector<int> classIds;
	vector<float> confidences;
	vector<Rect> boxes;

	for (size_t i = 0; i < outs.size(); ++i)
	{
		// Scan through all the bounding boxes output from the network and keep only the
		// ones with high confidence scores. Assign the box's class label as the class
		// with the highest score for the box.
		float* data = (float*)outs[i].data;
		for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
		{
			Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
			Point classIdPoint;
			double confidence;
			// Get the value and location of the maximum score
			minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
			if (confidence > confThreshold)
			{
				int centerX = (int)(data[0] * frame.cols);
				int centerY = (int)(data[1] * frame.rows);
				int width = (int)(data[2] * frame.cols);
				int height = (int)(data[3] * frame.rows);
				int left = centerX - width / 2;
				int top = centerY - height / 2;

				classIds.push_back(classIdPoint.x);
				confidences.push_back((float)confidence);
				boxes.push_back(Rect(left, top, width, height));
			}
		}
	}

	// Perform non maximum suppression to eliminate redundant overlapping boxes with
	// lower confidences
	vector<int> indices;
	NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
	for (size_t i = 0; i < indices.size(); ++i)
	{
		int idx = indices[i];
		Rect box = boxes[idx];
		drawPred(classIds[idx], confidences[idx], box.x, box.y,
			box.x + box.width, box.y + box.height, frame);
	}
}

// Remove the bounding boxes with low confidence using non-maxima suppression
vector<Rect> Yolo::getBoxes(Mat& frame, const vector<Mat>& outs)
{
	vector<int> classIds;
	vector<float> confidences;
	vector<Rect> boxes;

	for (size_t i = 0; i < outs.size(); ++i)
	{
		// Scan through all the bounding boxes output from the network and keep only the
		// ones with high confidence scores. Assign the box's class label as the class
		// with the highest score for the box.
		float* data = (float*)outs[i].data;
		for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
		{
			Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
			Point classIdPoint;
			double confidence;
			// Get the value and location of the maximum score
			minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
			if (confidence > confThreshold)
			{
				int centerX = (int)(data[0] * frame.cols);
				int centerY = (int)(data[1] * frame.rows);
				int width = (int)(data[2] * frame.cols);
				int height = (int)(data[3] * frame.rows);
				int left = centerX - width / 2;
				int top = centerY - height / 2;

				classIds.push_back(classIdPoint.x);
				confidences.push_back((float)confidence);
				boxes.push_back(Rect(left, top, width, height));
			}
		}
	}

	// Perform non maximum suppression to eliminate redundant overlapping boxes with
	// lower confidences
	vector<int> indices;
	vector<Rect> results;
	NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
	for (size_t i = 0; i < indices.size(); ++i)
	{
		int idx = indices[i];
		results.push_back(boxes[idx]);
	}

	return results;
}

// Draw the predicted bounding box
void Yolo::drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame)
{
	//Draw a rectangle displaying the bounding box
	rectangle(frame, Point(left, top), Point(right, bottom), Scalar(255, 178, 50), 3);

	//Get the label for the class name and its confidence
	string label = format("%.2f", conf);
	if (!classes.empty())
	{
		CV_Assert(classId < (int)classes.size());
		label = classes[classId] + ":" + label;
	}

	//Display the label at the top of the bounding box
	int baseLine;
	Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
	top = max(top, labelSize.height);
	rectangle(frame, Point(left, top - (int)round(1.5 * labelSize.height)), Point(left + (int)round(1.5 * labelSize.width), top + baseLine), Scalar(255, 255, 255), FILLED);
	putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 0), 1);
}

// Get the names of the output layers
vector<String> Yolo::getOutputsNames(const Net& net)
{
	static vector<String> names;
	if (names.empty())
	{
		//Get the indices of the output layers, i.e. the layers with unconnected outputs
		vector<int> outLayers = net.getUnconnectedOutLayers();

		//get the names of all the layers in the network
		vector<String> layersNames = net.getLayerNames();

		// Get the names of the output layers in names
		names.resize(outLayers.size());
		for (size_t i = 0; i < outLayers.size(); ++i)
			names[i] = layersNames[outLayers[i] - 1];
	}
	return names;
}