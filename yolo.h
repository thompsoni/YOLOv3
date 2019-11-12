#pragma once
#include <fstream>
#include <sstream>
#include <iostream>
#include <cstring>
#include <string>
#include <windows.h>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/types_c.h>

using namespace cv;
using namespace dnn;
using namespace std;

class Yolo
{

public:
	// Initialize the parameters
	float confThreshold = 0.3f; // Confidence threshold
	float nmsThreshold = 0.4f;  // Non-maximum suppression threshold
	//int inpWidth = 416;  // Width of network's input image
	//int inpHeight = 416; // Height of network's input image
	vector<string> classes;
	string classesFile = "files/coco.names";
	String modelConfiguration = "files/yolov3-tiny.cfg";
	String modelWeights = "files/yolov3-tiny.weights";
	string str;
	VideoCapture cap;
	VideoWriter video;
	Net net;
	//Mat frame, blob;

	int x_size;
	int y_size; // <-- Your res for the image
	HBITMAP hBitmap; // <-- The image represented by hBitmap
	Mat matBitmap; // <-- The image represented by mat
	HDC hdcSys;
	HDC hdcMem;
	void* ptrBitmapPixels;
	BITMAPINFO bi;
	HDC hdc;
	vector<Rect> faces;

	Yolo();
	void setup(int width, int height);
	void detect(Mat& img);

	// Remove the bounding boxes with low confidence using non-maxima suppression
	void postprocess(Mat& frame, const vector<Mat>& out);
	vector<Rect> getBoxes(Mat& frame, const vector<Mat>& out);

	// Draw the predicted bounding box
	void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame);

	// Get the names of the output layers
	vector<String> getOutputsNames(const Net& net);

};

