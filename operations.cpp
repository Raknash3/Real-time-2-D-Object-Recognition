/*
    * operations.cpp
    *
    *  Created on: Feb 20, 2024

    Authors:
    Name: Ravi Shankar Sankara Narayanan
    NUID: 001568628

    Name: Haritha Selvakumaran
    NUID: 002727950

    Pupose: This file contains all the function definitions that are used in the object recognition system.
            It is used in the main.cpp file to perform the object recognition system.

*/

#include "operations.h"
#include <opencv2/opencv.hpp>
#include <numeric>
#include <fstream>
#include <vector>
#include <utility>
#include <queue>
#include <string>

// Function to perform thresholding on an input image
cv::Mat performThresholding(const cv::Mat &inputImage)
{
    cv::Mat grayImage, blurredImage, thresholdedImage;

    // Convert the image to grayscale
    cv::cvtColor(inputImage, grayImage, cv::COLOR_BGR2GRAY);

    // Blur the image to reduce noise and detail
    cv::GaussianBlur(grayImage, blurredImage, cv::Size(5, 5), 0);

    // Perform thresholding to separate objects from the background
    cv::threshold(blurredImage, thresholdedImage, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    // Invert the colors (objects white, background black)
    cv::bitwise_not(thresholdedImage, thresholdedImage);

    return thresholdedImage;
}

// Function to perform K-Means thresholding on an input image (Not used in the project due to computational expense)
cv::Mat performKMeansThresholding(const cv::Mat &inputImage)
{
    cv::Mat grayImage, blurredImage, thresholdedImage, labels, centers;

    // Convert the image to grayscale
    cv::cvtColor(inputImage, grayImage, cv::COLOR_BGR2GRAY);

    // Blur the image to reduce noise and detail
    cv::GaussianBlur(grayImage, blurredImage, cv::Size(5, 5), 0);

    // Reshape the image to be a 1D array
    cv::Mat reshapedImage = blurredImage.reshape(1, blurredImage.total());

    // Convert to floating point for k-means
    reshapedImage.convertTo(reshapedImage, CV_32F);

    // Perform k-means clustering to find the most dominant colors
    cv::kmeans(reshapedImage, 2, labels, cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 1.0), 3, cv::KMEANS_PP_CENTERS, centers);

    // Get the mean value of the two centers
    double thresholdValue = (centers.at<float>(0, 0) + centers.at<float>(1, 0)) / 2.0;

    // Perform thresholding
    cv::threshold(blurredImage, thresholdedImage, thresholdValue, 255, cv::THRESH_BINARY);

    // Invert the colors (objects white, background black)
    cv::bitwise_not(thresholdedImage, thresholdedImage);

    return thresholdedImage;
}

// Function to perform morphological filtering on an input image
cv::Mat performMorphologicalFiltering(const cv::Mat &inputImage)
{
    cv::Mat morphImage;

    // Define the structuring element
    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));

    // Perform morphological operations to remove noise and small objects
    cv::morphologyEx(inputImage, morphImage, cv::MORPH_OPEN, element);
    cv::morphologyEx(morphImage, morphImage, cv::MORPH_CLOSE, element);

    return morphImage;
}

// Function to perform connected components analysis on an input image
cv::Mat performConnectedComponentsAnalysis(const cv::Mat &inputImage)
{
    cv::Mat labels, stats, centroids;

    // Perform connected components analysis to label each object
    int nLabels = cv::connectedComponentsWithStats(inputImage, labels, stats, centroids, 8, CV_32S, cv::CCL_WU);

    std::vector<cv::Vec3b> colors(nLabels);
    colors[0] = cv::Vec3b(0, 0, 0); // Background is black

    // Assign random color to each label (region)
    for (int label = 1; label < nLabels; ++label)
    {
        colors[label] = cv::Vec3b((rand() & 255), (rand() & 255), (rand() & 255));
    }

    // Create output image with different colors for each object
    cv::Mat outputImage(inputImage.size(), CV_8UC3);
    for (int y = 0; y < inputImage.rows; ++y)
    {
        for (int x = 0; x < inputImage.cols; ++x)
        {
            int label = labels.at<int>(y, x);
            outputImage.at<cv::Vec3b>(y, x) = colors[label];
        }
    }

    return outputImage;
}

// Function to compute the features of a region
RegionFeatures computeRegionFeatures(const cv::Mat &labels, const cv::Mat &stats, int regionId)
{
    RegionFeatures features;

    // Compute the axis-aligned bounding box
    features.boundingBox = cv::Rect(
        stats.at<int>(regionId, cv::CC_STAT_LEFT),
        stats.at<int>(regionId, cv::CC_STAT_TOP),
        stats.at<int>(regionId, cv::CC_STAT_WIDTH),
        stats.at<int>(regionId, cv::CC_STAT_HEIGHT));

    // Find the points in the region
    std::vector<cv::Point> points;
    for (int y = 0; y < labels.rows; ++y)
    {
        for (int x = 0; x < labels.cols; ++x)
        {
            if (labels.at<int>(y, x) == regionId)
            {
                points.push_back(cv::Point(x, y));
            }
        }
    }

    // Compute the oriented bounding box
    features.orientedBoundingBox = cv::minAreaRect(points);

    // Compute the percent filled
    int regionArea = stats.at<int>(regionId, cv::CC_STAT_AREA);
    double boundingBoxArea = features.orientedBoundingBox.size.area();
    features.percentFilled = static_cast<double>(regionArea) / boundingBoxArea;

    // Compute the bounding box ratio
    features.boundingBoxRatio = static_cast<double>(features.orientedBoundingBox.size.width) / features.orientedBoundingBox.size.height;

    // Compute the moments
    cv::Moments moments = cv::moments(labels == regionId);

    // Compute the central moments
    double mu20 = moments.mu20 / moments.m00;
    double mu02 = moments.mu02 / moments.m00;
    double mu11 = moments.mu11 / moments.m00;

    // Compute the axis of least central moment
    features.theta = 0.5 * std::atan2(2 * mu11, mu20 - mu02);

    return features;
}

// This function saves the features of a region and its associated label to a file.
void saveFeatureVector(const RegionFeatures &features, const std::string &label)
{
    // Open the file in append mode. If the file does not exist, it is created.
    // std::ofstream file("E:/MSCS/CVPR/projects/project3/data/objects/object_db.txt", std::ios::app);
    // std::ofstream file("E:/MSCS/CVPR/projects/project3/data/objects/object_db_task6.txt", std::ios::app);
    std::ofstream file("E:/MSCS/CVPR/projects/project3/data/objects/object_db_task9.txt", std::ios::app);

    // Write the label and the features to the file.
    file << label << " "
         << features.percentFilled << " "
         << features.boundingBoxRatio << " "
         << features.theta << " "
         << features.orientedBoundingBox.center.x << " "
         << features.orientedBoundingBox.center.y << " "
         << features.orientedBoundingBox.size.width << " "
         << features.orientedBoundingBox.size.height << " "
         << features.orientedBoundingBox.angle << "\n";
}

// This function loads the object database from a file.
std::vector<std::pair<RegionFeatures, std::string>> loadObjectDatabase()
{
    std::vector<std::pair<RegionFeatures, std::string>> objectDatabase;

    // Open the file
    // std::ifstream file("E:/MSCS/CVPR/projects/project3/data/objects/object_db.txt");
    // std::ifstream file("E:/MSCS/CVPR/projects/project3/data/objects/object_db_task6.txt");
    std::ifstream file("E:/MSCS/CVPR/projects/project3/data/objects/object_db_task9.txt");

    // Check if the file was opened successfully
    if (!file.is_open())
    {
        std::cerr << "Unable to open file objectDatabase.txt";
        return objectDatabase;
    }

    // Read the feature vectors and labels from the file
    std::string label;
    RegionFeatures features;
    while (file >> label >> features.percentFilled >> features.boundingBoxRatio >> features.theta >> features.orientedBoundingBox.center.x >> features.orientedBoundingBox.center.y >> features.orientedBoundingBox.size.width >> features.orientedBoundingBox.size.height >> features.orientedBoundingBox.angle)
    {
        objectDatabase.push_back(std::make_pair(features, label));
    }

    // Close the file
    file.close();

    return objectDatabase;
}

// This function calculates the scaled Euclidean distance between two feature vectors.
double calculateScaledEuclideanDistance(const RegionFeatures &a, const RegionFeatures &b)
{
    // Calculate the Euclidean distance
    double distance = std::sqrt(std::pow(a.percentFilled - b.percentFilled, 2) +
                                std::pow(a.boundingBoxRatio - b.boundingBoxRatio, 2) +
                                std::pow(a.theta - b.theta, 2) +
                                std::pow(a.orientedBoundingBox.center.x - b.orientedBoundingBox.center.x, 2) +
                                std::pow(a.orientedBoundingBox.center.y - b.orientedBoundingBox.center.y, 2) +
                                std::pow(a.orientedBoundingBox.size.width - b.orientedBoundingBox.size.width, 2) +
                                std::pow(a.orientedBoundingBox.size.height - b.orientedBoundingBox.size.height, 2) +
                                std::pow(a.orientedBoundingBox.angle - b.orientedBoundingBox.angle, 2));

    return distance;
}

/**
double calculateScaledEuclideanDistance(const RegionFeatures &a, const RegionFeatures &b, const RegionFeatures &stdev)
{
    // Calculate the Euclidean distance
    double distance = std::sqrt(std::pow((a.percentFilled - b.percentFilled) / stdev.percentFilled, 2) +
                                std::pow((a.boundingBoxRatio - b.boundingBoxRatio) / stdev.boundingBoxRatio, 2) +
                                std::pow((a.theta - b.theta) / stdev.theta, 2) +
                                std::pow((a.orientedBoundingBox.center.x - b.orientedBoundingBox.center.x) / stdev.orientedBoundingBox.center.x, 2) +
                                std::pow((a.orientedBoundingBox.center.y - b.orientedBoundingBox.center.y) / stdev.orientedBoundingBox.center.y, 2) +
                                std::pow((a.orientedBoundingBox.size.width - b.orientedBoundingBox.size.width) / stdev.orientedBoundingBox.size.width, 2) +
                                std::pow((a.orientedBoundingBox.size.height - b.orientedBoundingBox.size.height) / stdev.orientedBoundingBox.size.height, 2) +
                                std::pow((a.orientedBoundingBox.angle - b.orientedBoundingBox.angle) / stdev.orientedBoundingBox.angle, 2));

    return distance;
}
**/

// This function calculates the Manhattan distance between two feature vectors.
double calculateManhattanDistance(const RegionFeatures &a, const RegionFeatures &b)
{
    // Calculate the Manhattan distance
    double distance = std::abs(a.percentFilled - b.percentFilled) +
                      std::abs(a.boundingBoxRatio - b.boundingBoxRatio) +
                      std::abs(a.theta - b.theta) +
                      std::abs(a.orientedBoundingBox.center.x - b.orientedBoundingBox.center.x) +
                      std::abs(a.orientedBoundingBox.center.y - b.orientedBoundingBox.center.y) +
                      std::abs(a.orientedBoundingBox.size.width - b.orientedBoundingBox.size.width) +
                      std::abs(a.orientedBoundingBox.size.height - b.orientedBoundingBox.size.height) +
                      std::abs(a.orientedBoundingBox.angle - b.orientedBoundingBox.angle);

    return distance;
}

// This function calculates the cosine distance between two feature vectors.
double calculateCosineDistance(const RegionFeatures &a, const RegionFeatures &b)
{
    // Calculate the dot product
    double dotProduct = a.percentFilled * b.percentFilled +
                        a.boundingBoxRatio * b.boundingBoxRatio +
                        a.theta * b.theta +
                        a.orientedBoundingBox.center.x * b.orientedBoundingBox.center.x +
                        a.orientedBoundingBox.center.y * b.orientedBoundingBox.center.y +
                        a.orientedBoundingBox.size.width * b.orientedBoundingBox.size.width +
                        a.orientedBoundingBox.size.height * b.orientedBoundingBox.size.height +
                        a.orientedBoundingBox.angle * b.orientedBoundingBox.angle;

    // Calculate the magnitudes
    double magnitudeA = std::sqrt(std::pow(a.percentFilled, 2) +
                                  std::pow(a.boundingBoxRatio, 2) +
                                  std::pow(a.theta, 2) +
                                  std::pow(a.orientedBoundingBox.center.x, 2) +
                                  std::pow(a.orientedBoundingBox.center.y, 2) +
                                  std::pow(a.orientedBoundingBox.size.width, 2) +
                                  std::pow(a.orientedBoundingBox.size.height, 2) +
                                  std::pow(a.orientedBoundingBox.angle, 2));

    double magnitudeB = std::sqrt(std::pow(b.percentFilled, 2) +
                                  std::pow(b.boundingBoxRatio, 2) +
                                  std::pow(b.theta, 2) +
                                  std::pow(b.orientedBoundingBox.center.x, 2) +
                                  std::pow(b.orientedBoundingBox.center.y, 2) +
                                  std::pow(b.orientedBoundingBox.size.width, 2) +
                                  std::pow(b.orientedBoundingBox.size.height, 2) +
                                  std::pow(b.orientedBoundingBox.angle, 2));

    // Calculate the cosine similarity
    double cosineSimilarity = dotProduct / (magnitudeA * magnitudeB);

    // Calculate the cosine distance
    double cosineDistance = 1.0 - cosineSimilarity;

    return cosineDistance;
}

// This function finds the nearest neighbor in the object database for a given feature vector.
std::string findNearestNeighbor(const RegionFeatures &features, const std::vector<std::pair<RegionFeatures, std::string>> &objectDatabase)
{
    // Initialize the minimum distance to a large value
    double minDistance = std::numeric_limits<double>::max();
    std::string nearestNeighborLabel;

    // Iterate over the object database
    for (const auto &object : objectDatabase)
    {
        // Calculate the distance to the current object

        double distance = calculateScaledEuclideanDistance(features, object.first);
        // double distance = calculateManhattanDistance(features, object.first);
        // double distance = calculateCosineDistance(features, object.first);

        // If the distance is smaller than the current minimum, update the minimum and the nearest neighbor
        if (distance < minDistance)
        {
            minDistance = distance;
            nearestNeighborLabel = object.second;
        }
    }

    // Return the label of the nearest neighbor
    return nearestNeighborLabel;
}

/**
std::string findKNearestNeighbors(const RegionFeatures &features, const std::vector<std::pair<RegionFeatures, std::string>> &objectDatabase, int K)
{
    // Use a priority queue to store the K nearest neighbors
    // The queue is ordered by the distance, so the farthest neighbor is always at the top
    std::priority_queue<std::pair<double, std::string>> nearestNeighbors;

    // Iterate over the object database
    for (const auto &object : objectDatabase)
    {
        // Calculate the distance to the current object
        double distance = calculateScaledEuclideanDistance(features, object.first);

        // If we have less than K neighbors, we just add the current object to the queue
        if (nearestNeighbors.size() < K)
        {
            nearestNeighbors.push({distance, object.second});
        }
        else
        {
            // If the current object is closer than the farthest neighbor in the queue, we remove the farthest neighbor and add the current object
            if (distance < nearestNeighbors.top().first)
            {
                nearestNeighbors.pop();
                nearestNeighbors.push({distance, object.second});
            }
        }
    }

    // Now we have a queue with the K nearest neighbors
    // We create a map to count the occurrences of each label
    std::map<std::string, int> labelCounts;
    while (!nearestNeighbors.empty())
    {
        labelCounts[nearestNeighbors.top().second]++;
        nearestNeighbors.pop();
    }

    // We find the label with the most occurrences
    std::string mostCommonLabel;
    int maxCount = 0;
    for (const auto &labelCount : labelCounts)
    {
        if (labelCount.second > maxCount)
        {
            maxCount = labelCount.second;
            mostCommonLabel = labelCount.first;
        }
    }

    return mostCommonLabel;
}
**/

// This function finds the K nearest neighbors in the object database for a given feature vector.
std::string findKNearestNeighbors(const RegionFeatures &inputFeatures, const std::vector<std::pair<RegionFeatures, std::string>> &objectDatabase, int K)
{
    // Define a priority queue to store the K nearest neighbors
    auto comp = [](const std::pair<double, std::string> &a, const std::pair<double, std::string> &b)
    { return a.first < b.first; };

    std::priority_queue<std::pair<double, std::string>, std::vector<std::pair<double, std::string>>, decltype(comp)> nearestNeighbors(comp);

    // Iterate over the object database
    for (const auto &object : objectDatabase)
    {
        // Calculate the distance to the input features
        double distance = calculateScaledEuclideanDistance(inputFeatures, object.first);

        // If there are less than K neighbors in the queue, add the current object
        if (nearestNeighbors.size() < K)
        {
            nearestNeighbors.push({distance, object.second});
        }
        // If there are K or more neighbors, and the current object is closer than the farthest neighbor, replace the farthest neighbor
        else if (distance < nearestNeighbors.top().first)
        {
            nearestNeighbors.pop();
            nearestNeighbors.push({distance, object.second});
        }
    }

    // Calculate the weighted average of the labels of the K nearest neighbors
    double totalWeight = 0.0;
    std::map<std::string, double> labelWeights;
    while (!nearestNeighbors.empty())
    {
        double weight = 1.0 / nearestNeighbors.top().first; // Use the inverse of the distance as the weight
        labelWeights[nearestNeighbors.top().second] += weight;
        totalWeight += weight;
        nearestNeighbors.pop();
    }

    // Find the label with the highest average weight
    std::string highestWeightLabel;
    double highestWeight = -1.0;
    for (const auto &labelWeight : labelWeights)
    {
        double averageWeight = labelWeight.second / totalWeight;
        if (averageWeight > highestWeight)
        {
            highestWeight = averageWeight;
            highestWeightLabel = labelWeight.first;
        }
    }

    // Return the label with the highest average weight
    return highestWeightLabel;
}
