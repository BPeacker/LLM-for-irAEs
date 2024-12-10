# Detecting severe immune-related adverse events using large language models

**Repository for**: [Enhancing precision in detecting severe immune-related adverse events: comparative analysis of large language models and ICD codes in patient records](https://ascopubs.org/doi/abs/10.1200/JCO.24.00326?journalCode=jco)

**Authors**: Virginia H. Sun, MD; Julius C. Heemelaar, MD; Ibrahim Hadzic, MSc; Vineet K. Raghu, PhD; Chia-Yun Wu, MD; Leyre Zubiri, MD, PhD; Azin Ghamari, MD; Nicole R. LeBoeuf, MD, MPH; Osama Abu-Shawer, MD, MS; Kenneth L. Kehl, MD, MPH; Shilpa Grover, MD, MPH; Prabhsimranjot Singh, MD; Giselle A. Suero-Abreu, MD, PhD, MSc; Jessica Wu, BA; Ayo S. Falade, MD, MBA, APGD; Kelley Grealish, MSN, NP; Molly F. Thomas, MD, PhD; Nora Hathaway, MSN, NP; Benjamin D. Medoff, MD; Hannah K. Gilman, BS; Alexandra-Chloe Villani, PhD; Jor Sam Ho, MPH; Meghan J. Mooradian, MD; Meghan E. Sise, MD; Daniel A. Zlotoff, MD, PhD; Steven M. Blum, MD; Michael Dougan, MD, PhD; Ryan J. Sullivan, MD; Tomas G. Neilan, MD, MPH; and Kerry L. Reynolds, MD

**Code written by**: Virginia H. Sun, MD and Bryan L. Peaker

**Published in**: [Journal of Clinical Oncology](https://ascopubs.org/journal/jco/)

**Article citation**: Sun, V. H., Heemelaar, J. C., Hadzic, I., Raghu, V. K., Wu, C. Y., Zubiri, L., ... & Reynolds, K. L. (2024). Enhancing Precision in Detecting Severe Immune-Related Adverse Events: Comparative Analysis of Large Language Models and International Classification of Disease Codes in Patient Records. Journal of Clinical Oncology, JCO-24.

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

This repository contains the code and data used in the paper [Enhancing precision in detecting severe immune-related adverse events: comparative analysis of large language models and ICD codes in patient records](https://ascopubs.org/doi/abs/10.1200/JCO.24.00326?journalCode=jco). The paper presents a large language model (LLM) pipeline with Retrieval Augmented Generation (RAG) to detect immune-related adverse events (irAEs) among hospitalized patients. The main contributions of the paper are:
- LLMs can automatically detect irAEs with >90% sensitivity and specificity, outperforming ICD codes
- This automated tool can detect ICI colitis, hepatitis, pneumonitis, and myocarditis at a rate of <10s per chart
- RAG allows for greater transparency by providing relevant source-text used by the LLM to generate its response

We designed this tool using open-source architecture requiring minimal computational resources. Thus, this software is fully accessible and HIPAA-compliant when run on a local machine. Please cite our [original article](https://ascopubs.org/doi/abs/10.1200/JCO.24.00326?journalCode=jco) if you use this tool, and please feel free to contact us if you have any questions.

## Installation

1. Clone the repository:
    ```bash
    git clone [https://github.com/username/repo_name.git](https://github.com/ginnysun/LLM-for-irAEs)
    cd LLM-for-irAEs
    ```

2. Install Anaconda as according to your operating system:
- [Anaconda](https://docs.anaconda.com/anaconda/install/)

3. Install Ollama
- [Ollama](https://github.com/ollama/ollama)

## Usage

### Set up your Ollama server

1. Ensure Ollama was properly installed by running the following command. If not, follow the [installation instructions](https://github.com/ollama/ollama) as according to your operating system.
   ```bash
   ollama --version
   ```
2. Pull the model you want to use.
    ```bash
    ollama pull mistral-openorca
    ```
3. Start up your Ollama server. You may either run it directly using the Desktop application or run it in your shell. If using the latter, set up an Ollama server in a tmux session (on Linux/OSX) as demonstrated below. This allows you to make API requests to the Ollama server.
    ```bash
    tmux new -s ollama-server
    ollama serve
    ```
   Use ```Ctrl-B + d``` to detach from your session, and ```tmux attach-session -t ollama-server``` to return to your session as necessary.

### Set up your Anaconda environment

**Option 1: Creating the environment with required packages from the environment.yml file**

1. Create a new Anaconda environment from the ```environment.yml``` file:
    ```bash
    conda env create -f environment.yml
    conda activate demoenv
    ```
    This will create the environment and install the required packages based on those listed in environment.yml. Anaconda will install operating system-specific dependencies as needed.
2. Verify that the required packages were installed correctly:
    ```bash
    conda env export --from-history
    ```

    This will pull up a list of the installed packages. Verify that these packages are the same versions as the ones listed in the manual installation below.

**Option 2: Manually creating the environment and installing the required packages with Anaconda:**

1. Create a new Anaconda environment:
    ```bash
    conda create -n demoenv python=3.10
    conda activate demoenv
    ```
2. Install [Pytorch](https://pytorch.org/get-started/previous-versions/) (version 2.1.0) as according to your operating system.
- E.g. for OSX:
    ```bash
    conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 -c pytorch
    ```
3. Install the necessary packages into your new environment:
    ```bash
    conda install langchain=0.1.2 -c conda-forge
    conda install numpy pyreadr pandas tqdm
    conda install sentence-transformers=2.2.2
    conda install conda-forge::chromadb=0.4.22
    ```

### Running the Code

1. Follow the steps in ```./scripts/demo_LLM_walkthrough.pdf``` for a step-by-step guide, while ensuring everything runs smoothly. Note that you will have to create your own data to replace ```demo_reports.rdata``` (see [Data](#data) for how to format the file):

2. Consider editing ```demo_LLM_loop_noGPU.py``` based on any troubleshooting you needed to do while replicating the steps outlined in ```demo_LLM_walkthrough.pdf```. If all goes well, run the full analysis:
    ```bash
    python ./scripts/demo_LLM_loop_noGPU.py
    ```
   This code will output a csv file titled ```demo_LLM_loop_results.csv``` containing the LLM responses and corresponding source text retrieved via RAG. 

### Speeding up the code with GPU

1. This may be tricky. Be sure to know what type of GPU is being used. Ensure that Ollama is compatable with your GPU [here](https://github.com/ollama/ollama/blob/main/docs/gpu.md).

2. If using CUDA, install either NVCC Toolkit version [11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive) or [12.1](https://developer.nvidia.com/cuda-12-1-0-download-archive).

3. In your Anaconda environment (from above), download Pytorch v2.1.0 with the corresponding CUDA version.
    ```bash
    # CUDA 11.8
    conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
    # CUDA 12.1
    conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia
    ```

4. Ensure your Ollama server is using the GPU. This should be automatic, though if you come across any issues consider reading through the [troubleshooting guide](https://github.com/ollama/ollama/blob/main/docs/troubleshooting.md) for Ollama.

5. Similar to above, consider editing ```demo_LLM_loop_GPU.py``` based on any troubleshooting you may need to do. If all goes well, run the full analysis:
    ```bash
    python ./scripts/demo_LLM_loop_GPU.py
    ```
    
## Data

### Data Sources

Due to protected health information (PHI) included in the data sources, we are unable to provide the dataset used in this code. We will attempt to create sample data using simulated patients and will include it in the repository when available. 

The dataset used in this code, ```demo_reports.rdata``` contains the following information in this format:

| Patient_ID | Adjudicated_Case    | Text    |
| :---:   | :---: | :---: |
| 1 | Hepatitis   | progress note 1 text   |
| 1 | Hepatitis   | progress note 2 text   |
| ... | Hepatitis   | ...   |
| 1 | Hepatitis   | progress note 30 text   |
| 1 | Hepatitis   | discharge summary text   |
| 2 | N/A   | progress note 1 text   |
| ... | ...   | ...   |

Data description: 
- Patient_ID: Unique ID assigned to a patient's hospitalization
- Adjudicated_Case: Reference case as determined by manual adjudication
- Text: Raw text of progress notes and discharge summaries written during patient's hospitalization

### Preprocessing

Datasets used in this research were created using data from the Research Patient Data Registry at Massachusetts General Brigham. To create similar datasets at your institution, consider the following steps:  
1. Create a list of all inpatient hospital encounters of patients receiving immune checkpoint inhibitor therapy, containing the patient ID, admission date, and discharge date.
2. Filter the list of encounters to ensure the patient was admitted to the hospital AFTER receiving immune checkpoint inhibition therapy. Consider also filtering based on time since starting therapy (i.e. 6 months, 1 year).
3. Collect all progress notes, discharge summaries, and any other relevant notes written in the time frame of their hospitalization and store them in a RData file with a corresponding patient/hospitalization ID number. In our manuscript, we also included notes written the day before admission and up to five days after discharge to account for pre-admission notes and delays in provider notewriting.

## Repository Structure

```
repo_name/
│
├── modelfiles/           # Example modelfiles that can be loaded onto the Ollama server
├── scripts/              # Scripts for running the LLM
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
