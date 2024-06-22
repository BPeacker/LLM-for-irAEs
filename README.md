# Detecting severe immune-related adverse events using large language models

**Repository for**: [Enhancing precision in detecting severe immune-related adverse events: comparative analysis of large language models and ICD codes in patient records](link_to_paper)

**Authors**: [Author Names](link_to_author_profiles)

**Published in**: [Journal of Clinical Oncology](https://ascopubs.org/journal/jco/)

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
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

1. Clone the repository:
    ```bash
    git clone https://github.com/username/repo_name.git
    cd repo_name
    ```
    
2. Install Anaconda:
- [Anaconda](https://docs.anaconda.com/free/anaconda/install/)

3. Install packages into an Anaconda environment using the following command:
     ```bash
     conda create --name <ENV> --file requirements.txt
     ```

## Usage

### Set up your Ollama server

1. Ensure Ollama was properly installed by running the following command in your Anaconda environment. If not, follow the [installation instructions](https://github.com/ollama/ollama) as according to your operating system.
   ```bash
   ollama --version
   ```
2. (Optional) Load the Modelfile for the corresponding LLM into Ollama. The Modelfile included in this repository uses Mistral 7B Open Orca and sets the temperature parameter to zero and max token output to 256.
    ```bash
    ollama create example -f ./modelfiles/mistralopenorca_for_irAEs.Modelfile
    ```
3. Pull the model you want to use.
    ```bash
    ollama pull mistral-openorca ## OR ollama pull example, if you followed step 2. from above
    ```
4. Start up your Ollama server in a separate tmux session. This allows you to make API requests to the Ollama server while running Python code.
    ```bash
    tmux new -s ollama-server
    conda activate <ENV>
    ollama serve
    ```
   Use ```Ctrl-B + d``` to detach from your session, and ```tmux attach-session -t ollama-server``` to return to your session as necessary.

### Running the Code

1. Follow the steps in ```./scripts/demo_LLM_walkthrough.pdf``` for a step-by-step guide, while ensuring everything runs smoothly. Note that you will have to create your own data to replace ```demo_reports.rdata``` (see #data for how to format the file):
2. If all goes well, run the full analysis:
    ```bash
    python ./scripts/demo_LLM_loop.py
    ```
   This code will output a csv file titled ```demo_LLM_loop_results.csv``` containing the LLM responses and corresponding source text retrieved via RAG. 

## Data

### Data Sources

Due to protected health information (PHI) included in the data sources, we are unable to provide the dataset used in this code. We will attempt to create sample data using simulated patients and will include it in the repository when available. 

The dataset used in this code, ```demo_reports.rdata``` contains the following information in this format:

| Patient_ID | Case    | Text    |
| :---:   | :---: | :---: |
| 1 | Hepatitis   | progress note 1 text   |
| 1 | Hepatitis   | progress note 2 text   |
| ... | Hepatitis   | ...   |
| 1 | Hepatitis   | progress note 30 text   |
| 1 | Hepatitis   | discharge summary text   |
| 2 | N/A   | progress note 1 text   |
| ... | ...   | ...   |


### Preprocessing

Datasets used in this research were created using data from the Research Patient Data Registry at Massachusetts General Brigham. To create similar datasets at your institution, consider the following steps:  
1. Create a list of all inpatient hospital encounters of patients receiving immune checkpoint inhibitor therapy, containing the patient ID, admission date, and discharge date.
2. Filter the list of encounters to ensure the patient was admitted to the hospital AFTER receiving immune checkpoint inhibition therapy. Consider also filtering based on time since starting therapy (i.e. 6 months, 1 year).
3. Collect all progress notes, discharge summaries, and any other relevant notes written in the time frame of their hospitalization and store them in a RData file with a corresponding patient/hospitalization ID number. In our manuscript, we also included notes written the day before admission and up to five days after discharge to account for pre-admission notes and delays in provider notewriting.

## Repository Structure

Briefly describe the structure of the repository:

```
repo_name/
│
├── modelfiles/           # Example modelfiles that can be loaded onto the Ollama server
├── demo_scripts/         # Scripts for running the LLM
├── README.md             # This README file
├── requirements.txt      # Python dependencies
└── LICENSE               # License file
```

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

For any questions or issues, please contact [Ginny Sun, MD](vsun1@mgh.harvard.edu).

## Acknowledgements

- Research reported in this publication was supported by the National Heart, Lung, and Blood Institute of the National Institutes of Health under award number K24HL150238, as well as the Pugh Family for their generous donation to the Pugh Scholar Fund.
- We would additionally like to thank the Severe Immunotherapy Complications Service and the Cardiac Imaging Research Lab for their support.
