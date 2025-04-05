# Suzuka 2025 F1 Race Prediction

This repository contains a Python script and associated data files used to predict the outcomes of the Suzuka 2025 Formula 1 race. The project utilizes historical race data, driver performance metrics, and machine learning techniques to forecast the finishing positions, podium probabilities, and points finishes for each driver.

## Table of Contents

1. [Overview](#overview)
2. [Setup](#setup)
3. [Data](#data)
4. [Usage](#usage)
5. [Visualizations](#visualizations)
6. [Contributing](#contributing)
7. [License](#license)

## Overview 

The project aims to predict the results of the Suzuka 2025 Formula 1 race by analyzing historical data and applying statistical models. The script generates various visualizations to help understand the predictions and compare driver performances.

## Setup

To run the script and generate the predictions, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Nephyedge/F1--Preduction-2025.git
   cd F1--Preduction-2025
   cd Suzuka

2. Data
    Historical race data is loaded dynamically from the FastF1 library.
    Output is saved to suzuka_2025_predictions.csv

3. Usage
    ```bash
    python suzuka_f1.py

4. Visualizations
    The generated visualizations provide insights into the predicted race outcomes:
        Main Race Prediction Chart: Shows predicted finishing positions with error bars indicating uncertainty.
        Podium Probability Chart: Displays the top drivers by their probability of finishing on the podium.
        Grid vs Predicted Position Comparison: Compares grid positions with predicted race finishes.
        Points Probability Chart: Shows the probability of each driver scoring points.
        Driver Head-to-Head Comparison: Compares key metrics between two top drivers.
        Team Performance Overview: Summarizes the predicted performance of each team.
        Prediction Uncertainty Chart: Visualizes the uncertainty in predictions for each driver.

5. Contributing
    Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

6. License
    This project is licensed under the MIT License. See the LICENSE file for details.
