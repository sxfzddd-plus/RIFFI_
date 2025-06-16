📄 Description
The Python scripts in this repository contain the code used to produce the experimental results presented in the paper:
“Regional Isolation Forest: A Novel Approach for Enhanced Anomaly Detection and Interpretability.”

📁 Repository Structure
models/
Contains the Python scripts that implement the models proposed in the paper, including the Regional Isolation Forest (RIF) and its feature importance method RIFFI.

utils/
Includes utility scripts that define helper functions used within the model classes.

performance/
Provides testing scripts for evaluating the proposed methods.

RIF_demo.py: Anomaly detection test script that includes experiments on three synthetic datasets and public UCL datasets.

RIFFI_demo.py: Interpretability test script offering evaluation examples using UCL datasets.

🧩 Dependencies
The main Python libraries used in this project are listed below. Please ensure compatible versions are installed for reproducibility of results:

Library	Version
numpy	2.2.3
scikit-learn	1.6.1
pandas	2.2.3
scipy	1.15.2

You can install the dependencies using:

bash
pip install numpy==2.2.3 scikit-learn==1.6.1 pandas==2.2.3 scipy==1.15.2