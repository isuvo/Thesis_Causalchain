# Causal Analysis of Software Vulnerabilities

This project aims to build a framework for analyzing software vulnerabilities using causal inference techniques. By representing code as a Code Property Graph (CPG) and leveraging program slicing, we can identify causal chains that lead to vulnerabilities.

## Project Structure

- `notebooks/`: Jupyter notebooks for exploration, training, and demonstration.
- `data/`: Contains all data, from raw datasets to processed training samples.
- `external/`: Houses external tools like Joern.
- `models/`: Stores trained model weights and checkpoints.
- `src/`: Source code for data preprocessing, model training, inference, and utilities.
- `requirements.txt`: Python dependencies.
- `README.md`: This file.

## Getting Started

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Setup External Tools:**
    - Download and install Joern in the `external/joern` directory or add it to your system's PATH.

3.  **Run the Pipeline:**
    - Follow the scripts in the `src/` directory to process data, train a model, and perform causal analysis.
