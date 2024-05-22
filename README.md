# Traffic Violation Detection System

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Algorithms Used](#algorithms-used)
  - [Red Light Violation Detection](#red-light-violation-detection)
  - [Over Speeding Detection](#over-speeding-detection)
  - [License Plate Recognition](#license-plate-recognition)
- [Setup and Installation](#setup-and-installation)
  - [Prerequisites](#prerequisites)
  - [Installation Steps](#Setup-and-Installation)
- [Usage](#usage)
- [Screenshots](#screenshots)
- [Patent and Research Paper](#patent-and-research-paper)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)
- [Additional Resources](#additional-resources)

## Introduction

The Traffic Violation Detection System is an advanced, AI-driven solution designed to enhance road safety by automatically detecting and reporting traffic violations. The system focuses on two primary violations: red light violations and over-speeding. Utilizing state-of-the-art image processing and machine learning techniques, the system captures real-time footage, detects violations, and extracts vehicle license plate information for enforcement.

This project has filed a patent for its innovative algorithms and published a research paper detailing its methodology and results.

## Features

- **Red Light Violation Detection**: Identifies and captures vehicles running red lights.
- **Over Speeding Detection**: Detects and captures vehicles exceeding speed limits.
- **License Plate Recognition**: Extracts and reads license plates of violating vehicles.
- **High Accuracy**: Utilizes advanced algorithms for precise detection and recognition.
- **Real-Time Processing**: Capable of detecting violations and generating alerts in real-time.

## System Architecture

The system comprises the following main components:

1. **Image/Video Capture Module**: Captures real-time video footage of traffic intersections and road segments.
2. **Violation Detection Module**: Processes the captured footage to detect red light violations and over-speeding instances.
3. **License Plate Recognition Module**: Extracts and reads license plates of vehicles identified as violating traffic rules.
4. **Alert and Reporting Module**: Generates alerts and detailed reports for each detected violation, including images and license plate information.

## Algorithms Used

### Red Light Violation Detection

1. **Traffic Light State Recognition**:
    - Uses Convolutional Neural Networks (CNNs), specifically a custom-trained model based on YOLOv5, to detect the state of traffic lights (red, yellow, green).
    - Processes video frames to monitor the state of traffic lights continuously.
    - Utilizes techniques like non-maximum suppression to eliminate redundant bounding boxes.

2. **Vehicle Detection and Tracking**:
    - Employs YOLOv5 for real-time vehicle detection with high accuracy and speed.
    - Tracks vehicles across frames using the SORT (Simple Online and Realtime Tracking) algorithm, which assigns unique IDs to detected vehicles and maintains their trajectories.

3. **Violation Confirmation**:
    - Analyzes vehicle positions when the light turns red.
    - Confirms a violation if a vehicle crosses a predefined virtual stop line during the red light state.
    - Uses frame differencing and background subtraction to accurately determine vehicle movement.

### Over Speeding Detection

1. **Speed Calculation**:
    - Tracks vehicle movement between frames using the SORT algorithm.
    - Calculates speed using the formula:

      \[
      \text{Speed} = \frac{\Delta \text{Distance}}{\Delta \text{Time}}
      \]

      where \(\Delta \text{Distance}\) is calculated based on pixel distance converted to real-world distance using camera calibration parameters, and \(\Delta \text{Time}\) is derived from the frame rate.
    - Utilizes homography transformation to map pixel coordinates to real-world coordinates accurately.

2. **Threshold Comparison**:
    - Compares the calculated speed against predefined speed limits.
    - Flags vehicles exceeding the speed limit for over-speeding violations.
    - Implements a moving average filter to smooth speed measurements and reduce false positives.

### License Plate Recognition

1. **License Plate Detection**:
    - Utilizes OpenCV for initial license plate localization.
    - Applies edge detection techniques (Canny edge detection) and contour analysis to identify potential license plate regions.
    - Filters out non-license plate regions using aspect ratio and geometric properties.

2. **Character Recognition**:
    - Applies Optical Character Recognition (OCR) techniques to read the license plate characters.
    - Uses the Tesseract OCR engine, pre-trained on alphanumeric data, for initial recognition.
    - Enhances accuracy with a custom-trained CNN model for character recognition, fine-tuned on a dataset of license plates.
    - Implements image preprocessing steps like binarization, noise reduction, and character segmentation to improve OCR performance.

## Setup and Installation

### Prerequisites

- Python 3.8 or higher
- OpenCV
- TensorFlow/Keras
- YOLOv8 model weights
- Tesseract OCR

Certainly! Here is the detailed setup or installation part to be included in the README file for your Traffic Violation Detection System:

## 🚀 Setup and Installation

Follow these steps to set up and run the Traffic Violation Detection System on your local machine:

### 1. Clone the Repository

First, clone the repository to your local machine using the following command:

```bash
git clone https://github.com/shrey1010/traffic-violation-detection-system.git
cd traffic-violation-detection-system
```

### 2. Set Up a Virtual Environment

It is recommended to create a virtual environment to manage your dependencies. You can create a virtual environment using `venv`:

```bash
python -m venv venv
```

Activate the virtual environment:

- On Windows:
  ```bash
  venv\Scripts\activate
  ```
- On macOS and Linux:
  ```bash
  source venv/bin/activate
  ```

### 3. Install Dependencies

Install the required Python libraries listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

This command will install all necessary libraries such as OpenCV, TensorFlow, PyTorch, YOLOv5, Tesseract OCR, and other dependencies needed for the project.

### 4. Install YOLOv5

Since YOLOv5 requires specific setup steps, follow these instructions to install YOLOv5:

```bash
# Navigate to the YOLOv5 directory
cd yolov5

# Install YOLOv5 requirements
pip install -r requirements.txt

# Go back to the root directory
cd ..
```

Ensure you follow any additional setup instructions provided in the [YOLOv5 repository](https://github.com/ultralytics/yolov5).

### 5. Download Pre-trained Models

Download the pre-trained models required for vehicle detection and license plate recognition. Place these models in the designated directory within the project structure. If specific URLs or sources are provided for the models, follow those instructions.

### 6. Configure Tesseract OCR

Ensure Tesseract OCR is installed on your system. You can download and install Tesseract from [here](https://github.com/tesseract-ocr/tesseract).

After installation, update the `pytesseract.pytesseract.tesseract_cmd` path in your code to point to the Tesseract executable:

```python
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update this path
```

### 7. Run the System

With all dependencies installed and configured, you can now run the Traffic Violation Detection System. Use the following command to start the detection process:

```bash
python Traffic_Red-Light Running_Violation_Detection.ipynb
```

Make sure to pass any necessary arguments or configure the system settings as required in your project documentation or code comments.

### 8. Viewing Results

After running the system, results including detected violations and recognized license plates will be saved in the output directory. Review the output images and logs to verify the results.


## Usage

1. **Configure Video Source**:
    - Update the `config.json` file with the appropriate video source (live feed or pre-recorded video).

2. **Start Detection**:
    - Run the main script to start detecting traffic violations.
    - The system will process the video feed, detect violations, and extract license plate information.

3. **View Reports**:
    - The system generates reports for each detected violation, including images and license plate details.
    - Access these reports from the `output/` directory.

## Screenshots

*Below are screenshots demonstrating the system's output:*

### Red Light Violation Detection
![Red Light Violation](screenshots/green_light.png)
![Red Light Violation](screenshots/yellow_light.png)
![Red Light Violation](screenshots/red_light.png.png)
![Red Light Violation](screenshots/yellow_light.png)
![Red Light Violation](screenshots/yellow_light.png)

### Over Speeding Detection
![Over Speeding Detection](screenshots/over_speeding.png.png)

### License Plate Recognition
![License Plate Recognition](screenshots/no_plate.png.png)

## Patent and Research Paper

The algorithms and methodologies used in this project are patent-pending. The implementation details and results are documented in our research paper.

- **Patent**: [Patent Number (Filed)](https://docs.google.com/document/d/1JaS3gQIQxGgdCy3_bWtRPeC93M4OEjjHFZEtsPx__p0/edit?usp=drivesdk)
- **Research Paper**: [Download the Research Paper](https://docs.google.com/document/d/1Kr72VyfM8-1ljIy2fa-111rS2ELVDeUG0NgIuInafxo/edit)

## Contributing

We welcome contributions from the community.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For any queries or further information, please contact:

- **Email**: shreyshukla1010@gmail.com
- **GitHub**: [https://github.com/shrey1010/traffic-violation-detection-system](https://github.com/shrey1010/traffic-violation-detection-system)

## Additional Resources

- 🌐 **Kaggle Notebook**: Interested in a Kaggle environment? Explore the notebook [here](https://www.kaggle.com/code/farzadnekouei/traffic-red-light-running-violation-detection).
- 📹 **Input Video Data**: Access the raw and modified video [here](https://www.kaggle.com/datasets/farzadnekouei/license-plate-recognition-for-red-light-violation).

---

Thank you for using the Traffic Violation Detection System! Your feedback and contributions are highly appreciated.