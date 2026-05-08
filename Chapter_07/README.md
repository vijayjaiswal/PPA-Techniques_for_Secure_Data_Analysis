# Chapter 7: Hands-On with SDV and CTGAN

This directory contains the implementation of synthetic data generation using the **Synthetic Data Vault (SDV)** ecosystem and the **CTGAN** model.

## Overview

The exercise demonstrates how to:
1.  **Generate Mock Data**: Creates a simulated customer transaction log with various data types (Numerical, Categorical, Datetime).
2.  **Define Metadata**: Uses SDV's `Metadata` class to automatically detect and refine the data schema.
3.  **Train CTGAN**: Initializes and trains a Conditional Tabular GAN (CTGAN) model on the real data.
4.  **Sample Synthetic Records**: Generates new, privacy-preserving synthetic records that maintain the statistical properties of the original dataset.
5.  **Evaluate Quality**: Runs built-in SDV evaluation tools to compare column distributions and correlations between real and synthetic datasets.

## Setup Instructions

1.  **Install Dependencies**:
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the Tutorial**:
    ```bash
    python sdv_ctgan_tutorial.py
    ```

## Key Components

-   `sdv_ctgan_tutorial.py`: The main script containing the end-to-end workflow.
-   `requirements.txt`: List of necessary libraries (`sdv`, `pandas`, `numpy`).

## Interpretation of Results

-   **Quality Score**: A high-level indicator of how well the synthetic data mimics the real data. Closer to 100% is better.
-   **Column Shapes**: Measures how well the marginal distributions of individual columns match.
-   **Diagnostic Report**: Confirms that the synthetic data respects the structure and constraints of the metadata.
