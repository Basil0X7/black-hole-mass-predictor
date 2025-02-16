# Black Hole Mass Predictor

## Summary

This project explores the use of machine learning models to predict the mass of black holes using data from the Sloan Digital Sky Survey's (SDSS) Data Release 16 (DR16) catalog. By leveraging key parameters such as FWHM (Full Width at Half Maximum) and luminosity measurements from various spectral lines, the model provides an accurate and efficient alternative to manual calculations of black hole mass. Four models were employed in this study: Linear Regression, Lasso Regression, Ridge Regression, and Random Forest Regression.

## Introduction

The accurate estimation of black hole mass is fundamental for understanding their role in the evolution of galaxies and the dynamics of the universe. Current methods for calculating black hole mass often involve complex equations and manual computation, which can be prone to errors and are time-consuming. This research aims to address these challenges by employing machine learning techniques to predict black hole mass efficiently.

## Dataset

The dataset was sourced from the Sloan Digital Sky Survey's (SDSS) Data Release 16 (DR16) catalog, a highly reliable and peer-reviewed astronomical dataset. The DR16 catalog contains detailed measurements of quasars, stars, and galaxies. For this study, we focused on the following key columns:

- **FWHM (Hβ, Mg II, C IV)**: These represent the Full Width at Half Maximum values for different emission lines, providing crucial information about the broadening of spectral lines due to the velocity of gas near the black hole.
  
- **LOGL5100, LOGL3000, LOGL1350**: Luminosities at rest-frame wavelengths of 5100 Å, 3000 Å, and 1350 Å, respectively. These luminosities are used to estimate the accretion rate and indirectly infer black hole mass.
  
- **LOGMBH**: The fiducial single-epoch black hole mass, which serves as the target variable for prediction.
  
- **Z_DR16Q**: Redshift values, essential for understanding the cosmological distance and the quasar's properties.
  
- **SNR Columns (Signal-to-Noise Ratios)**: These include GAIA_G_FLUX_SNR, GAIA_BP_FLUX_SNR, which provide quality indicators of the measured fluxes from the Gaia mission.

These parameters were chosen due to their direct correlation with the black hole mass and their significance in established astrophysical models. The data underwent rigorous cleaning and preprocessing to handle missing or erroneous values.

## Methodology

### Preprocessing

The data underwent extensive cleaning, including handling missing values, scaling, and feature extraction.

### Models

Four machine learning models were utilized:

- Linear Regression
- Lasso Regression
- Ridge Regression
- Random Forest Regression

### Evaluation Metrics

- Mean Squared Error (MSE)
- R-squared (R²)

## Results

The following table summarizes the performance of the four models on validation and test datasets:

| Model                   | Validation MSE | Test MSE | Validation R² | Test R² |
|-------------------------|----------------|----------|---------------|---------|
| Linear Regression        | 178,842,702,216.48 | 978,836,000,265.01 | -688,948,050,950.83 | -3,766,974,617,741.10 |
| Lasso Regression         | 1,285.84 | 0.13 | -4,952.40 | 0.48 |
| Ridge Regression         | 11,263,002,362,089.08 | 61,644,294,698,663.45 | -43,387,979,655,073.81 | -237,233,298,934,253.94 |
| Random Forest Regression | 0.07 | 0.07 | 0.74 | 0.74 |

### Feature Importance Analysis

Using Random Forest Regression, we evaluated the relative importance of each feature in the prediction. Features such as FWHM (Mg II and Hβ) and luminosity (LOGL5100 and LOGL3000) exhibited the strongest correlation with the target variable (LOGMBH). The Signal-to-Noise Ratio (SNR) parameters showed less significant impact but were retained to ensure model robustness.

The correlation between features and their predictive influence suggests that emission line properties and luminosities remain the most critical predictors of black hole mass, aligning with established astrophysical theories.

### Negative Values in Model Metrics

It is important to note the occurrence of negative values in the **R²** and **MSE** metrics for some models, particularly Linear, Ridge, and Lasso Regression. These negative values are likely due to the models underfitting the data, which means they are not capturing the underlying patterns effectively. In regression analysis, R² values can be negative when the model performs worse than simply predicting the mean value of the target variable (LOGMBH). Similarly, high MSE values and negative R² indicate poor model performance in terms of predicting black hole mass.

The poor performance of these models can be attributed to several factors:

1. **Feature Engineering**: The chosen features may not fully capture the complexity of the relationship between the input variables and the black hole mass, leading to suboptimal performance for simpler models like Linear and Ridge Regression.
2. **Model Complexity**: More complex models, such as Random Forest Regression, are better equipped to capture non-linear relationships and interactions between features, explaining why they outperformed the other models.
3. **Hyperparameter Tuning**: The models might require further tuning of their hyperparameters to achieve optimal performance.

These observations highlight the need for further refinement in feature selection, model choice, and hyperparameter tuning for more accurate predictions.

## Data Verification Formulas

To ensure the accuracy and consistency of the dataset, I used the following astrophysical formulas to cross-check the data:

## Data Verification Formulas

To ensure the accuracy and consistency of the dataset, the following astrophysical formulas were used to cross-check the data:

#### **Formula 1:**

The first formula provides a theoretical estimation of the black hole mass based on the Full Width at Half Maximum (FWHM) of the Hβ emission line and the luminosity at 5100 Å:

![image](https://github.com/user-attachments/assets/2c5fae6d-4b69-4e0a-a88c-34b3362eb1da)

Where:
- \( M_{\text{BH}} \) is the black hole mass.
- \( FWHM_{\text{H}\beta} \) is the Full Width at Half Maximum of the Hβ emission line.
- \( L_{5100} \) is the luminosity at 5100 Å.

This formula relates the black hole mass \( M_{\text{BH}} \) to the spectral width and luminosity, providing a way to estimate the mass.

#### **Formula 2:**

The second formula uses a logarithmic approach to estimate the black hole mass. It is based on empirical constants \( A \), \( B \), and \( C \) derived from observational data:

![image](https://github.com/user-attachments/assets/34b659da-20b6-4230-8d66-b27a15b9711f)

Where:
- \( M_{\text{BH}} \) is the black hole mass.
- \( FWHM \) is the Full Width at Half Maximum of the emission lines.
- \( L \) is the luminosity (usually \( L_{5100} \)).

This formula uses logarithmic scaling for both the FWHM and luminosity, with constants \( A \), \( B \), and \( C \) based on empirical data to match observed trends.
## Usage

1. Install Libraries
```bash
pip install -r requirements.txt
```
2. Run the Program
```bash
uvicorn main:app --reload
```
3. Click [http://127.0.0.1:8000](http://127.0.0.1:8000)

## Conclusion

The implementation of machine learning for black hole mass prediction provides a promising pathway for improving efficiency and accuracy in astrophysical studies. Random Forest Regression outperformed other models in this study, achieving the lowest MSE and highest R² scores. Future work will involve refining the models, expanding the dataset, and incorporating additional astrophysical features to enhance predictive power.

## Acknowledgment

I would like to thank Dr. Raid Suleiman and Fatma Yousef from the Atomic and Molecular Physics Division, Center for Astrophysics | Harvard & Smithsonian, for their invaluable support and guidance in preparing this paper.

