# SWDB 2025 Project to understand underlying functional-structural connectivity using the Allen Institute V1 Deep dive dataset:

<img align = right height="200" src="https://github.com/user-attachments/assets/d0ff038e-8d39-4cd7-8141-6b4260c6a968" />

Members:

- Arun Cherkkil 
- Libby Zhang
- Rodrigo Gonzalez Laiz

Description: We study the relationship between function and structure in  the V1DD dataset by examining single cell and population level neural activity. We would like to examine how the neural correlations differ across conditions and what functional and structural motifs underlie their organization.

Dataset Used: V1 Deep Dive (V1DD). Single cell activity measured across a cubic millimeter of the mouse visual cortex while the animal is headfixed and exposed to a variety of visual cues including gratings, low sparse noise , natural images and movies. 

# Population coupling analysis to identify functional classes in calcium dataset

Run the following notebook to generate the population coupling analysis plots:

**Main Analysis Notebook:** [code/scripts/2025-09-03-p_coupling_AC.ipynb](./code/scripts/2025-09-03-p_coupling_AC.ipynb)

This notebook performs population coupling analysis to identify "choristers" (cells highly coupled to population activity) vs "soloists" (cells with independent activity patterns).

## Key Results

### Population Coupling Distribution
![Population Coupling](./images/population_coupling.png)

Distribution of population coupling values across all cells, showing the separation between choristers and soloists.

### Correlation Heatmaps
![Sorted Heatmap](./images/sorted_heatmap.png)

Sorted correlation matrix showing clear block structure when cells are organized by their population coupling strength.

![Sorted vs Unsorted](./images/sorted_vs_unsorted.png)

Comparison of correlation matrices before and after sorting by population coupling.

### PCA and Eigenspectrum Analysis
![PCA Analysis](./images/PCA.png)

Principal component analysis showing the dimensionality of neural population activity.

![Eigenspectrum](./images/eigen.png)

Eigenspectrum analysis revealing the effective dimensionality of the neural population.

### Cell Class Comparisons
![Chorister PCA](./images/chorister_PCA.png)
![Soloist PCA](./images/soloists_PCA.png)

PCA analysis separately for chorister and soloist populations.

### Heatmap showing cell cell correlation for chositers and soloists 

![CS Heatmap](./images/sorted_cs_heatmap.png)

![Mean Activity vs Population Coupling](./images/meanactivity_vs_pcoupling.png)

Relationship between mean neural activity and population coupling strength.

### Stimulus-Specific Responses
![DFF Traces](./images/DFF_traces_all_epochs.png)

ΔF/F traces across different stimulus epochs (drifting gratings, natural movies, spontaneous activity).

![Average DFF](./images/average_dff_across_stimulus.png)

Average ΔF/F responses across different stimulus conditions.

# Additional Analyses

## Chorister vs Soloist Structural Analysis

**Notebook:** [scripts/choroist_vs_soloists.ipynb](./scripts/choroist_vs_soloists.ipynb)

This notebook examines the structural connectivity patterns underlying functional differences between choristers and soloists, including distance analysis and logistic regression modeling.

## Null Models for Structural Connectivity

**Notebook:** [scripts/null_models_structural.ipynb](./scripts/null_models_structural.ipynb)

Statistical validation using null models to assess the significance of observed structural connectivity patterns.

