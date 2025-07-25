# Automatic Software Requirements Generation from User Reviews (Thesis Project)

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Research](https://img.shields.io/badge/Research-Thesis%20Project-orange.svg)]()

![Workflow Diagram](https://github.com/RayM11/Requirements-Generation-from-user-reviews/blob/main/Docs/detailed-workflow-diagram.png)
*Figure: High-level workflow of the solution, from user reviews to unified requirements.*

## 📜 Project Overview

This repository contains the source code and documentation for my undergraduate thesis project for the degree of Computer Engineer: **"Automatic Software Requirements Generation from User Reviews combining Machine Learning and Large Language Models (LLMs)"**.

The primary goal of this project is to address the challenge of processing massive volumes of user reviews generated on digital platforms. While valuable, this feedback often has a low density of useful information, making manual analysis unfeasible.

The proposed solution is a desktop application that implements a **3-phase automated pipeline** to transform raw user reviews into a consolidated, actionable list of software requirements, serving as a powerful tool for **Data-Driven Requirements Engineering (DDRE)**.

## ✨ Methodology

The system orchestrates a robust and modular process divided into three core stages:

1.  **Phase 1: Comment Filtering (Classification)**
    *   Uses a fine-tuned **Transformer (BERTweet)** model for the binary classification of reviews into "informative" and "non-informative".
    *   Implements a novel **domain knowledge injection** technique using feature vectors (RC - Relevant Count & RP - Relevant Position) based on the ISO/IEC/IEEE 24765 standard. This enhances the model's ability to identify feedback relevant to software development.
    *   **Core module:** `FilteringModule.py`

2.  **Phase 2: Comment Clustering**
    *   The filtered "informative" comments are grouped into thematic clusters to identify recurring issues or suggestions.
    *   The **Fuzzy C-means** algorithm is employed to handle ambiguity and allow a single comment to belong to multiple topics.
    *   **UMAP** is applied for dimensionality reduction on the generated embeddings to improve clustering quality.
    *   **Core module:** `ClusteringModule.py`

3.  **Phase 3: Requirements Generation (LLM)**
    *   Utilizes a **Large Language Model (LLM) such as DeepSeek** to generate software requirements from the comment clusters.
    *   This is a two-step process:
        1.  **Initial Generation:** Functional (FR) and Non-Functional (NFR) requirements are generated for each cluster individually.
        2.  **Unification & Consolidation:** A second, more complex prompt instructs the LLM to analyze all initial requirements, identify duplicates, merge similar ideas, and produce a final, clean, and consolidated list.
    *   **Core module:** `GenerationModule.py`

## 🚀 Key Features

*   **Fully Automated Pipeline:** From loading a CSV of user reviews to exporting a unified list of requirements.
*   **Intuitive Graphical User Interface (GUI):** Developed with **CustomTkinter** to facilitate easy experimentation and system use.
*   **High-Efficacy Classification:** Outperforms state-of-the-art solutions due to the domain knowledge injection technique.
*   **Quality Requirement Generation:** Produces structured requirements, classifies them as Functional or Non-Functional, and traces them back to the source comments.
*   **Robust Software Architecture:**
    *   **Singleton & Mediator Patterns:** The `RequirementsController` centralizes the workflow, ensuring a single instance and low coupling between modules.
    *   **N-Tier & Pipes and Filters Architectures:** The design separates presentation (GUI), logic (controller, modules), and model access, promoting reusability and maintainability.
*   **Highly Parametrizable:** The GUI allows users to configure models, algorithms, and key parameters for each stage of the pipeline.
*   **Report Generation:** Creates detailed reports and a final summary in a dedicated directory for each run.

## 🛠️ Tech Stack

*   **Core Language:** Python 3.X
*   **Graphical User Interface (GUI):** `customtkinter`
*   **Machine Learning & NLP:**
    *   `PyTorch` & `PyTorch Lightning`
    *   `Transformers (Hugging Face)`
    *   `scikit-learn` & `scikit-fuzzy`
    *   `umap-learn`
    *   `NLTK`
    *   `LangChain`
*   **Data Handling & Numerics:** `pandas`, `numpy`
*   **Data Visualization:** `matplotlib`, `seaborn`, `plotly`

## 📊 Research Results

### Performance Highlights

- **Classification**: BERTweet-base + RC knowledge vector achieved best results
- **Clustering**: UMAP dimensionality reduction to 2 components showed optimal performance
- **Generation**: LLM-generated requirements showed high consistency with source opinions
- **Quality**: Over 50% of generated requirements met all evaluated quality properties

### Comparative Analysis

- Outperformed existing state-of-the-art solutions in user opinion classification
- Demonstrated significant improvement with domain knowledge integration
- Showed effective requirement consolidation through dual-LLM approach


## 🚀 Getting Started

### Prerequisites

```bash
# Python 3.8 or higher
pip install -r requirements.txt
```

### Basic Usage

#### Command Line Interface

```python
from Code.logic.controller.RequirementsController import RequirementsController

# Initialize controller
controller = RequirementsController.get_instance()

# Generate requirements
results = controller.generate_requirements_from_csv(
    csv_path="user_reviews.csv",
    app_description="Your application description",
    output_directory="./output",
    classification_model="BERTweet - base",
    clustering_algorithm="fuzzy c-means",
    llm_model="deepseek-ai/DeepSeek-V3-0324"
)
```
## ▶️ How to Use the Application

1.  **Launch the Application:**
    Run the `main.py` script from the root of the repository to start the GUI.
    ```bash
    python Code/main.py
    ```
2.  **Configure the Parameters:**
    *   Use the `📂` buttons to select your input CSV file (e.g., `user_comments.csv`) and the desired output directory.
    *   Adjust the parameters for the Filtering, Clustering, and Generation phases as needed.
3.  **Start the Process:**
    *   Click the `🚀 Start Generation` button.
    *   Progress and logs will be displayed in the interface's console.
4.  **View the Results:**
    *   Upon completion, a timestamped folder will be created in your output directory, containing the reports from each phase and the final requirements list.


## 📁 Project Structure

```
Code/
├── data/
│   ├── datasets/          # Sample datasets
│   └── glossary/          # Software engineering terminology
├── logic/
│   ├── commentFiltering/  # Opinion classification components
│   ├── commentClustering/ # Clustering algorithms and embeddings
│   ├── requirementsGeneration/ # LLM-based generation
│   └── controller/        # Main orchestrator and reporting
├── GUI/
│   └── RequirementsGeneratorGUI.py  # User interface
└── main.py
```

## 👨‍💻 Author

*   **Ray Maestre Peña**
*   📧 **Email:** raymaestrepena@gmail.com
*   💼 **LinkedIn:** ()

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.