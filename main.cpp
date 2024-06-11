/*
    * main.cpp
    *
    *  Created on: Feb 20, 2024

    Authors:
    Name: Ravi Shankar Sankara Narayanan
    NUID: 001568628

    Name: Haritha Selvakumaran
    NUID: 002727950

    Pupose: This file is the main file that runs the object recognition system.
            It uses the operations.h file to perform the object recognition system and
            implements the  functions defined in operations.cpp. The main program reads the input image,
            applies thresholding, morphological filtering, connected components analysis, and object recognition.

*/

#include "operations.h"
#include <opencv2/opencv.hpp>
#include <chrono>
#include <iomanip>
#include <ctime>

int main()
{
    // Open the default camera
    cv::VideoCapture cap(0);

    // Initialize the confusion matrix
    std::vector<std::vector<int>> confusionMatrix(5, std::vector<int>(5, 0));

    // Map the labels to indices
    std::map<std::string, int> labelToIndex;
    labelToIndex["bull_statue"] = 0;
    labelToIndex["pen"] = 1;
    labelToIndex["smartwatch"] = 2;
    labelToIndex["headphone"] = 3;
    labelToIndex["scissors"] = 4;

    // Check if camera opened successfully
    if (!cap.isOpened())
    {
        return -1;
    }

    cv::namedWindow("Original Video", 1);
    cv::namedWindow("Thresholded Video", 1);

    // Commenting this line because K-means is computationally expensive
    // cv::namedWindow("K-Means Thresholded Video", 1);

    cv::namedWindow("Morphological Filtered Video", 1);
    cv::namedWindow("Connected Components Analysis", 1);

    // Load the object database that has the feature vectors and labels
    std::vector<std::pair<RegionFeatures, std::string>> objectDatabase = loadObjectDatabase();

    // Check if the database loaded correctly
    std::cout << "Object Database Loaded" << std::endl;
    std::cout << "Size of objectDatabase: " << objectDatabase.size() << std::endl;

    for (;;)
    {
        cv::Mat frame;
        cap >> frame; // Get a new frame from camera

        // Perform thresholding operation
        cv::Mat thresholdedFrame = performThresholding(frame);

        // Perform k-means thresholding operation
        // cv::Mat thresholdedFrame = performKMeansThresholding(frame);

        // Perform morphological filtering operation
        cv::Mat morphFrame = performMorphologicalFiltering(thresholdedFrame);

        // Perform connected components analysis
        cv::Mat labels, stats, centroids;
        int nLabels = cv::connectedComponentsWithStats(morphFrame, labels, stats, centroids, 8, CV_32S, cv::CCL_WU);

        // Perform connected components analysis
        cv::Mat connectedComponentsFrame = performConnectedComponentsAnalysis(morphFrame);

        // Listen for a key press
        char key = cv::waitKey(30);

        for (int label = 1; label < nLabels; ++label)
        {
            if (stats.at<int>(label, cv::CC_STAT_AREA) < 100)
                continue; // Ignore small regions

            RegionFeatures features = computeRegionFeatures(labels, stats, label);

            // Draw the bounding box
            cv::rectangle(connectedComponentsFrame, features.boundingBox, cv::Scalar(0, 255, 0), 2);

            // Display the percent filled
            std::string percentFilledText = "Percent Filled: " + std::to_string(features.percentFilled);
            cv::putText(connectedComponentsFrame, percentFilledText, cv::Point(10, 30 + 60 * (label - 1)), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);

            // Display the bounding box ratio
            std::string boundingBoxRatioText = "Bounding Box Ratio: " + std::to_string(features.boundingBoxRatio);
            cv::putText(connectedComponentsFrame, boundingBoxRatioText, cv::Point(10, 50 + 60 * (label - 1)), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);

            // Display the axis of least central moment
            std::string thetaText = "Axis of Least Central Moment: " + std::to_string(features.theta);
            cv::putText(connectedComponentsFrame, thetaText, cv::Point(10, 70 + 60 * (label - 1)), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);

            // Draw the oriented bounding box
            cv::Point2f vertices[4];
            features.orientedBoundingBox.points(vertices);
            for (int i = 0; i < 4; ++i)
            {
                cv::line(connectedComponentsFrame, vertices[i], vertices[(i + 1) % 4], cv::Scalar(0, 255, 0), 2);
            }

            // N or n for adding the object to object database
            if (key == 'N' || key == 'n')
            {
                // Prompt the user for a label
                std::string userLabel;
                std::cout << "Enter a label for the current object: ";
                std::cin >> userLabel;

                // Clear the input stream
                std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

                // Save the feature vector and label
                saveFeatureVector(features, userLabel);

                // Reload the object database
                objectDatabase = loadObjectDatabase();
            }
            // key r for object recognition
            else if (key == 'r')
            {
                // Find the nearest neighbor in the object database
                std::string nearestNeighborLabel = findNearestNeighbor(features, objectDatabase);

                // Get the current time
                auto now = std::chrono::system_clock::now();
                std::time_t now_c = std::chrono::system_clock::to_time_t(now);
                std::cout << "Label: " << nearestNeighborLabel << ", Time: " << std::put_time(std::localtime(&now_c), "%F %T") << std::endl;

                // Display the label of the nearest neighbor
                std::string nearestNeighborText = "Nearest Neighbor: " + nearestNeighborLabel;
                cv::putText(connectedComponentsFrame, nearestNeighborText, cv::Point(10, 90 + 60 * (label - 1)), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
            }
            // key k for KNN recognition
            else if (key == 'k' || key == 'K')
            {
                // Define the number of neighbors to consider
                int K = 3;

                // Find the K nearest neighbors in the object database
                std::string knnLabel = findKNearestNeighbors(features, objectDatabase, K);

                // Get the current time
                auto now = std::chrono::system_clock::now();
                std::time_t now_c = std::chrono::system_clock::to_time_t(now);

                // Print the label of the K nearest neighbors
                std::cout << "KNN Label: " << knnLabel << ", Time: " << std::put_time(std::localtime(&now_c), "%F %T") << std::endl;
            }
            // Press key 'e' to enter evaluation mode
            else if (key == 'e')
            {
                // Find the nearest neighbor in the object database
                std::string predictedLabel = findNearestNeighbor(features, objectDatabase);

                // Prompt the user for the true label
                std::cout << "Predicted label: " << predictedLabel << ". Is this correct? (y/n): ";
                std::string response;
                std::cin >> response;

                std::string trueLabel;
                if (response == "y" || response == "Y")
                {
                    trueLabel = predictedLabel;
                }
                else
                {
                    std::cout << "Enter the correct label: ";
                    std::cin >> trueLabel;
                }

                int trueIndex = labelToIndex[trueLabel];
                int predictedIndex = labelToIndex[predictedLabel];

                // Increment the corresponding cell in the confusion matrix
                confusionMatrix[trueIndex][predictedIndex]++;
            }
        }

        // Display the original video
        cv::imshow("Original Video", frame);

        // Display the thresholded video
        cv::imshow("Thresholded Video", thresholdedFrame);

        // Display the k-means thresholded video
        // cv::imshow("K-Means Thresholded Video", thresholdedFrame);

        // Display the morphological filtered video
        cv::imshow("Morphological Filtered Video", morphFrame);

        // Display the connected components analysis
        cv::imshow("Connected Components Analysis", connectedComponentsFrame);

        // Break the loop on pressing 'q'
        if (key >= 0 && key == 'q')
            break;
    }

    // Print the confusion matrix
    std::cout << "\nConfusion Matrix:\n";
    for (const auto &row : confusionMatrix)
    {
        for (int cell : row)
        {
            std::cout << cell << ' ';
        }
        std::cout << '\n';
    }

    // The camera will be deinitialized
    return 0;
}