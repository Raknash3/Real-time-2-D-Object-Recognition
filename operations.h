/*
    * operations.h
    *
    *  Created on: Feb 19, 2024

    Authors:
    Name: Ravi Shankar Sankara Narayanan
    NUID: 001568628

    Name: Haritha Selvakumaran
    NUID: 002727950

    Pupose: This file contains the function declarations and the struct definition for the object recognition system.

*/

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

// Function to perform thresholding on an input image
cv::Mat performThresholding(const cv::Mat &inputImage);

// Function to perform K-Means thresholding on an input image
cv::Mat performKMeansThresholding(const cv::Mat &inputImage);

// Function to perform morphological filtering on an input image
cv::Mat performMorphologicalFiltering(const cv::Mat &inputImage);

// Function to perform connected components analysis on an input image
cv::Mat performConnectedComponentsAnalysis(const cv::Mat &inputImage);

// Struct to hold the features of a region
struct RegionFeatures
{
    cv::Rect boundingBox;                // Bounding box of the region
    double percentFilled;                // Percentage of the bounding box that is filled
    double boundingBoxRatio;             // Ratio of the width to the height of the bounding box
    double theta;                        // Axis of least central moment
    cv::RotatedRect orientedBoundingBox; // Oriented bounding box of the region
};

// Function to compute the features of a region
RegionFeatures computeRegionFeatures(const cv::Mat &labels, const cv::Mat &stats, int regionId);

// Function to save a feature vector with a given label
void saveFeatureVector(const RegionFeatures &features, const std::string &label);

// Function to calculate the scaled Euclidean distance between two feature vectors
double calculateScaledEuclideanDistance(const RegionFeatures &features1, const RegionFeatures &features2);

// Function to find the nearest neighbor in the object database for a given feature vector
std::string findNearestNeighbor(const RegionFeatures &features, const std::vector<std::pair<RegionFeatures, std::string>> &objectDatabase);

// Function to load the object database
std::vector<std::pair<RegionFeatures, std::string>> loadObjectDatabase();

// Function to calculate the Manhattan distance between two feature vectors
double calculateManhattanDistance(const RegionFeatures &a, const RegionFeatures &b);

// Function to find the K nearest neighbors in the object database for a given feature vector
std::string findKNearestNeighbors(const RegionFeatures &features, const std::vector<std::pair<RegionFeatures, std::string>> &objectDatabase, int K);