# Multi-Agent Manufacturing Inspection

A multi-agent, LLM-driven inspection framework for autonomous manufacturing quality assessment. The system integrates large language models, vision-language models, multimodal process data, and domain-specific knowledge to detect, interpret, and classify manufacturing anomalies.

## Overview

This project investigates the use of agentic AI for automated inspection in manufacturing processes. The framework combines specialized agents for visual inspection, sensor-data analysis, and review/reasoning.

The system is designed to process synchronized manufacturing data such as thermal imagery, electrical signals, geometric measurements, and process metadata. It supports multiple prompting and reasoning strategies, including zero-shot prompting, few-shot prompting, multimodal chain-of-thought reasoning, and memory-enhanced workflows.

## Key Features

- Multi-agent manufacturing inspection architecture
- Large language model and vision-language model reasoning
- Thermal image analysis
- Voltage and current signal analysis
- Domain-rule-guided inspection
- Zero-shot and few-shot prompting
- Multimodal chain-of-thought reasoning
- Memory-enhanced agent workflows
- Automated reviewer feedback
- Accuracy and performance evaluation
- Image-processing utilities for manufacturing inspection

## Repository Structure

    multi-agent-manufacturing-inspection/
    │
    ├── training/
    │   ├── images/
    │   │   ├── frame_difference.py
    │   │   └── updated_meta_data_to_image.py
    │   │
    │   ├── prompts/
    │   ├── reviewer_logs/
    │   ├── results/
    │   ├── results_accuracy_plot/
    │   ├── dspy_optimized_prompts/
    │   │
    │   ├── accuracy_plot.py
    │   ├── accuracy_plot_merged_frames.py
    │   ├── distill_lessons.py
    │   ├── dspy_prompt_optimize.py
    │   ├── extract_dspy_prompts.py
    │   ├── graph.py
    │   ├── inspect_checkpoints.py
    │   ├── main_memory.py
    │   ├── main_new.py
    │   ├── metrics_logger.py
    │   ├── plot_memory_routes.py
    │   ├── pre_processing_sensor_data.py
    │   ├── review_agent.py
    │   ├── review_agent_memory.py
    │   ├── tools.py
    │   ├── tools_memory.py
    │   ├── vision_agent_new.py
    │   ├── vision_agent_memory.py
    │   └── visualize_memory.py
    │
    ├── .gitignore
    ├── README.md
    └── requirement.txt

## Data

The complete manufacturing dataset is not included in this repository because the thermal-image and sensor datasets are large research files.

Large image and signal datasets are excluded from version control. Selected Python source files located inside `training/images/` are intentionally retained in the repository:

    training/images/frame_difference.py
    training/images/updated_meta_data_to_image.py

These scripts support image preprocessing and analysis while the large underlying image datasets remain excluded.

## System Workflow

The framework follows a multi-agent inspection workflow:

1. Manufacturing process data are collected and preprocessed.
2. Image-based information is prepared for visual inspection.
3. Vision agents analyze manufacturing images and process characteristics.
4. Sensor information is processed from synchronized process measurements.
5. Domain-specific manufacturing rules support interpretation and classification.
6. Review agents evaluate intermediate predictions and identify potential reasoning or classification errors.
7. Memory-enhanced workflows retain and reuse useful inspection knowledge across experiments.
8. Final predictions are compared with reference labels using automated evaluation metrics.

## Reasoning Strategies

The project supports multiple experimental reasoning configurations, including:

- Zero-shot prompting
- Few-shot prompting
- Zero-shot multimodal chain-of-thought prompting
- Few-shot multimodal chain-of-thought prompting
- Memory-enhanced agent reasoning
- Reviewer-guided refinement
- Prompt optimization workflows

## Multimodal Inspection

The framework is designed to reason over multiple manufacturing data modalities, including:

- Thermal images
- Melt-pool-related visual information
- Arc voltage
- Arc current
- Process metadata
- Geometric measurements
- Other synchronized manufacturing sensor information

The combination of visual, sensor, and domain-specific information enables the agents to perform manufacturing inspection using multimodal reasoning rather than relying on a single data source.

## Image Processing Utilities

The repository includes selected image-processing utilities under `training/images/`.

### frame_difference.py

Supports frame-difference-based image analysis for detecting temporal changes across manufacturing image sequences.

### updated_meta_data_to_image.py

Supports processing and integration of manufacturing metadata with corresponding image information.

The large raw image datasets used by these scripts are intentionally excluded from the repository.

## Agent Components

### Vision Agent

The vision agent analyzes image-derived manufacturing information and generates category-level inspection predictions.

Relevant files include:

    training/vision_agent_new.py
    training/vision_agent_memory.py

### Review Agent

The review agent evaluates predictions, identifies potential errors, and provides structured feedback that can be used to improve subsequent inspection decisions.

Relevant files include:

    training/review_agent.py
    training/review_agent_memory.py

### Memory-Enhanced Workflow

Memory-based versions of the system are used to investigate whether previously identified inspection knowledge can improve future reasoning.

Relevant files include:

    training/main_memory.py
    training/tools_memory.py
    training/vision_agent_memory.py
    training/review_agent_memory.py
    training/visualize_memory.py

## Prompt Optimization

The repository contains utilities for prompt optimization and lesson extraction.

Relevant files include:

    training/dspy_prompt_optimize.py
    training/extract_dspy_prompts.py
    training/distill_lessons.py
    training/dspy_optimized_prompts/

These components support experimentation with improved prompts and reusable reasoning knowledge.

## Evaluation

The framework includes utilities for evaluating inspection performance and visualizing experimental results.

Relevant files include:

    training/metrics_logger.py
    training/accuracy_plot.py
    training/accuracy_plot_merged_frames.py
    training/results/
    training/results_accuracy_plot/

Evaluation is used to compare different prompting, reasoning, and memory configurations.

## Installation

Clone the repository:

    git clone https://github.com/Supriya-Sarker/multi-agent-manufacturing-inspection.git
    cd multi-agent-manufacturing-inspection

Create a virtual environment:

    python -m venv venv

Activate the virtual environment on macOS or Linux:

    source venv/bin/activate

Install the required Python packages:

    pip install -r requirement.txt

## Running the Project

The primary experimental workflow is located inside the `training` directory.

Navigate to the training directory:

    cd training

Run the main inspection workflow:

    python main_new.py

For the memory-enhanced workflow:

    python main_memory.py

Additional scripts in the repository are used for preprocessing, evaluation, visualization, prompt optimization, and analysis.

Because the complete research dataset is not included in this repository, some scripts require local manufacturing image, sensor, or metadata files before they can be executed successfully.

## Research Applications

This project supports research in areas including:

- Agentic AI
- Autonomous manufacturing inspection
- Multimodal artificial intelligence
- Vision-language models
- Large language models for engineering applications
- Defect and anomaly detection
- Manufacturing process monitoring
- Explainable AI for manufacturing
- Automated quality assessment
- Intelligent manufacturing systems

## Current Status

This repository represents an active research project. The inspection architecture, prompting strategies, memory mechanisms, evaluation methods, and supporting utilities may continue to evolve as additional experiments are conducted.

## Author

**Supriya Sarker**  
Ph.D. Candidate
University of Arkansas, Fayetteville
United States
