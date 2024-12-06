# Words2Contact: Identifying Support Contacts from Verbal Instructions Using Foundation Models

![GitHub](media/concept_figure_wide.png)

Official implementation of the paper *"Words2Contact: Identifying Support Contacts from Verbal Instructions Using Foundation Models"* presented at IEEE-RAS Humanoids 2024.

This repository contains the implementation of the LLMs/VLMs part of the project. For the multi-contact whole-body controller, please contact the authors.

For more details, visit the [paper website](https://hucebot.github.io/words2contact_website/).

---

## Table of Contents
1. [Repository Structure](#repository-structure)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Usage](#usage)
    - [Quick Start](#quick-start)
    - [Command-Line Options](#command-line-options)
5. [Citation](#citing-words2contact-preprint)
6. [Acknowledgements](#acknowledgements)
7. [Contact](#contact)

---

## Repository Structure

```plaintext
.
├── .ci/                       # Docker configurations
│   └── Dockerfile             # Dockerfile to build the project's container
├── config/                    # Configuration files for models
│   └── GroundingDINO_SwinT_OGC.py # GroundingDINO configuration
├── data/                      # Test data and outputs
│   ├── test.png               # Example input image
│   └── test_output.png        # Example output image
├── media/                     # Media assets
│   ├── ack.png                # Acknowledgment image
│   └── concept_figure_wide.png # Conceptual figure for the project
├── submodules/                # External submodules
│   └── CLIP_Surgery/          # CLIP Surgery code and resources
├── words2contact/             # Core project source code
│   ├── grammar/               # Grammars for constraining language models
│   │   ├── classifier.gbnf    # Grammar for classifying outputs
│   │   └── README.md          # Grammar module documentation
│   ├── prompts/               # Prompts for LLMs
│   │   └── prompts.json       # JSON file with pre-defined prompts
│   ├── geom_utils.py          # Utilities for geometric calculations
│   ├── math_pars.py           # Parsing mathematical expressions
│   ├── saygment.py            # Language-grounded segmentation
│   ├── words2contacts.py      # Core script for Words2Contact
│   └── yello.py               # Language-grounded object detection
├── main.py                    # Entry point for the project
├── launch.sh                  # Docker launch script
├── object_detection.py        # Object detection testing
├── object_segmentation.py     # Object segmentation testing
└── README.md                  # Documentation (this file)
```

---

## Prerequisites
Before starting, ensure you have the following:
- **Docker** (v20.10 or later)
- **Python 3.10** or later (for running scripts outside Docker)
- An **OpenAI API Key** (if using GPT-based LLMs). You can obtain it from [OpenAI](https://platform.openai.com/).

---

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/hucebot/words2contact.git
    cd words2contact
    ```

2. Build the Docker image:
    ```bash
    docker build -t words2contact -f .ci/Dockerfile .
    ```

---

## Usage

### Set Up
If you plan to use OpenAI's GPT-based LLMs, set your API key as an environment variable before launching the Docker container:
```bash
export OPENAI_KEY=<your_openai_api_key>
```

### Launching the Docker Container
Run the following command to start the container:
```bash
bash launch.sh
```
This will create a `models/` folder in the root of the project where models will be downloaded and stored.

### Quick Start
To test Words2Contact with the provided example image:
```bash
python main.py --image_path data/test.png --prompt "Place your hand above the red bowl."
```
The output will be saved as `data/test_output.png`.

More examples coming soon!

### Command-Line Options
```plaintext
usage: main.py [-h] [--image_path IMAGE_PATH] [--prompt PROMPT] [--use_gpt] [--yello_vlm YELLO_VLM] [--output_path OUTPUT_PATH] [--llm_path LLM_PATH] [--chat_template CHAT_TEMPLATE]

Run Words2Contact with an image and a text prompt.

options:
  -h, --help                    show this help message and exit
  --image_path IMAGE_PATH       Path to the input image file. Default: 'data/test.png'.
  --prompt PROMPT               Text prompt for Words2Contact. Default: 'Place your hand above the red bowl.'.
  --use_gpt                     Use OpenAI API for the LLM (requires `OPENAI_KEY`).
  --yello_vlm YELLO_VLM         Model to use for YELLO VLM. Default: 'GroundingDINO'.
  --output_path OUTPUT_PATH     Path to save the output image. Default: 'data/test_output.png'.
  --llm_path LLM_PATH           Path to the `.gguf` LLM model weights.
  --chat_template CHAT_TEMPLATE Chat template to use for local LLMs. Default: 'ChatML'.
```

### Using Local LLMs
1. Download `.gguf` weights for local LLMs from a trusted source (e.g., [TheBloke's Hugging Face models](https://huggingface.co/TheBloke)).
2. Place the weights in the `models/` folder.
3. Specify the `--llm_path` argument when running the script:
    ```bash
    python main.py --image_path data/test.png --llm_path models/local_model.gguf
    ```

---

## Contact
For questions or support, please contact:
- **Dionis Totsila**: [dionis.totsila@inria.fr](mailto:dionis.totsila@inria.fr)

---

## Citing Words2Contact (preprint)
If you use Words2Contact in your research, please cite our paper:
```bibtex
@INPROCEEDINGS{totsila2024words2contactidentifyingsupportcontacts,
    author={Dionis Totsila and Quentin Rouxel and Jean-Baptiste Mouret and Serena Ivaldi},
    booktitle={2024 IEEE-RAS 23rd International Conference on Humanoid Robots (Humanoids)},
    title={Words2Contact: Identifying Support Contacts from Verbal Instructions Using Foundation Models},
    year={2024},
}
```

[**Read the preprint here**](https://arxiv.org/abs/XXXX.XXXXX) *(Add this link when available.)*

---

## Acknowledgements
This research was supported by:
- CPER CyberEntreprises
- Creativ’Lab platform of Inria/LORIA
- EU Horizon project euROBIN (GA n.101070596)
- France 2030 program through the PEPR O2R projects AS3 and PI3 (ANR-22-EXOD-007, ANR-22-EXOD-004)

<div align="center">
    <img src="media/ack.png" alt="Acknowledgments" width="50%">
</div>
