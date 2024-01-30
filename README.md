# **DRL-Based Decision Support**

Welcome to the Control Room Simulation Dataset repository! This dataset contains valuable information derived from a simulated formaldehyde production plant, involving participant interaction within a controlled experimental setting resembling a control room. The dataset is utilized for statistical analysis to compare outcomes among different groups and has potential applications for decision-makers involved in control room design and optimization, process safety engineers, system engineers, human factors engineers in process industries, and researchers in related domains. This repository contains the code for DRL-based decision support tool usage in control room scenarios: 

https://arrow.tudublin.ie/schfsehart/533/

## Overview
The dataset encompasses measurements obtained from diverse data sources during the period spanning May to August 2023. It incorporates both objective and subjective metrics commonly employed in assessing cognitive states related to workload, situational awareness, stress, and fatigue. Various data collection tools such as health monitoring watch, eye tracking, process and HMI logs, operational metrics (response time, reaction time, performance, etc), NASA Task Load Index (NASA-TLX), Situation Awareness Rating Technique (SART), a think-aloud Situation Presence Assessment Method (SPAM), AI support questions, and AI vs human error were utilized.

## Data Collection
Participants tested three scenarios lasting 15–18 minutes, with breaks and survey completion periods in between, utilizing different combinations of decision support tools. The decision support tools varied across groups, encompassing factors of digitized screen-based procedures and the inclusion of an AI recommendation system.

## Dataset Structure
The collected raw data was processed particularly for the analyses in the associated research paper. The data from individual participants was concatenated and merged into a single xls file for further evaluation. The data used for comparison between the GroupN and GroupAI is presented in the merged normalized data folder. The xls file contains the data points for each participant per row, and each column represents the data and sub-data collected from various sources. As the focus of this analysis is not between the scenarios, the data is normalized across scenarios to avoid the effect of the scenarios in the analysis and is averaged for every participant to acquire a single vector of data for each participant.

## Value of the Data
The dataset provides an opportunity to study the integration of human-in-the-loop configurations with AI systems in safety-critical industries. By examining the data, researchers can identify the factors necessary for successful collaboration between humans and AI. This knowledge can lead to the development of optimized interaction mechanisms, ensuring that the strengths of both humans and AI are leveraged effectively to enhance decision-making in critical scenarios.

## Accessing the Dataset
The dataset files are organized in the following structure:
- `merged_normalized_data.xls`: Contains the merged and normalized data for comparison between GroupN and GroupAI.
- `hmm_modeling_concatenated_data.csv`: Represents the time-series data of the process, alarms, and HMI logs for every participant into a single file as required by the HMM python library.
- `failed_participants_labels.csv`: Provides the labels for participants who failed during the task based on various factors such as the consequence of plant shutdown or reactor overheating and overall performance.

## Citation
If you use this dataset in your research or publication, please cite the associated research paper.

We hope this dataset proves to be valuable for your research and analysis. If you have any questions or need further assistance, feel free to reach out to the repository maintainers. Thank you for your interest in the Control Room Simulation Dataset!
