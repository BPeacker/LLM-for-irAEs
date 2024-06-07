# Detecting severe immune-related adverse events using large language models

**Repository for**: [Enhancing precision in detecting severe immune-related adverse events: comparative analysis of large language models and ICD codes in patient records](link_to_paper)

**Authors**: [Author Names](link_to_author_profiles)

**Published in**: [Journal Name](link_to_journal)

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Reproducing Results](#reproducing-results)
- [Repository Structure](#repository-structure)
- [License](#license)
- [Contact](#contact)
- [Acknowledgements](#acknowledgements)

## Introduction

This repository contains the code and data used in the paper [Enhancing precision in detecting severe immune-related adverse events: comparative analysis of large language models and ICD codes in patient records](link_to_paper). The paper presents a large language model (LLM) pipeline with Retrieval Augmented Generation (RAG) to detect immune-related adverse events (irAEs) among hospitalized patients. The main contributions of the paper are:
- LLMs can automatically detect irAEs with >90% sensitivity and specificity, outperforming ICD codes
- This automated tool can detect ICI colitis, hepatitis, pneumonitis, and myocarditis at a rate of <10s per chart
- RAG allows for greater transparency by providing relevant source-text used by the LLM to generate its response

We designed this tool using open-source architecture requiring minimal computational resources. Thus, this software is fully accessible and HIPAA-compliant when run on a local machine. Please cite our [original article](link to paper) if you use this tool, and please feel free to contact us if you have any questions.

## Installation

To run the code in this repository, you will need to install the following dependencies:

- [Ollama](https://github.com/ollama/ollama)
- [Anaconda](https://docs.anaconda.com/free/anaconda/install/)

You can install packages into an Anaconda environment using the following command:

```bash
conda create --name <env> --file requirements.txt
```

## Usage

### Set up your Ollama server

1. Follow the [installation instructions](https://github.com/ollama/ollama) as according to your operating system.
2. Load the Modelfile for the corresponding LLM into Ollama. The Modelfile included in this repository uses Mistral 7B Open Orca and sets the temperature parameter to zero and max token output to 256.
    ```bash
    ollama create example -f mistralopenorca_for_irAEs.Modelfile
    ```

### Running the Code

1. Clone the repository:
    ```bash
    git clone https://github.com/username/repo_name.git
    cd repo_name
    ```

2. Set up your Ollama server:
    ```bash
    python preprocess.py
    ```

3. Run the main analysis:
    ```bash
    python main_analysis.py
    ```

### Example

Provide an example of how to use the code. For instance:

```bash
python example.py --input data/input_file.csv --output results/output_file.csv
```

## Data

### Data Sources

Describe the datasets used in the paper, including where and how they can be accessed:

- Dataset 1: [Link to dataset](link)
- Dataset 2: [Link to dataset](link)

### Preprocessing

Explain any preprocessing steps required before running the analysis:

1. Step 1
2. Step 2

## Reproducing Results

To reproduce the results presented in the paper, follow these steps:

1. Download the datasets and place them in the `data` directory.
2. Run the preprocessing script:
    ```bash
    python preprocess.py
    ```
3. Execute the main analysis script:
    ```bash
    python main_analysis.py
    ```
4. The results will be saved in the `results` directory.

## Repository Structure

Briefly describe the structure of the repository:

```
repo_name/
│
├── data/                 # Raw and processed data
├── scripts/              # Scripts for preprocessing and analysis
├── results/              # Output results
├── notebooks/            # Jupyter notebooks for exploratory analysis
├── README.md             # This README file
├── requirements.txt      # Python dependencies
└── LICENSE               # License file
```

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

For any questions or issues, please contact [Author Name](link_to_email_or_profile).

## Acknowledgements

Acknowledge any collaborators, funding sources, or third-party libraries:

- Collaborator 1
- Funding source 1
- Library 1
