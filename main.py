from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import numpy as np
import joblib

app = FastAPI()
model = joblib.load("models/random_forest_regression.pkl")

class PredictionRequest(BaseModel):
    Z_DR16Q: float
    LOGMBH_HB: float
    LOGMBH_MGII: float
    LOGMBH_CIV: float
    LOGMBH: float
    LOGL5100: float
    LOGL3000: float
    LOGL1350: float
    HBETA_Peak_Wavelength: float
    HBETA_50_Percent_Flux_Centroid_Wavelength: float
    HBETA_Flux: float
    HBETA_Logarithm_of_Line_Luminosity: float
    HBETA_FWHM: float
    HBETA_REW: float
    MGII_Peak_Wavelength: float
    MGII_50_Percent_Flux_Centroid_Wavelength: float
    MGII_Flux: float
    MGII_Logarithm_of_Line_Luminosity: float
    MGII_FWHM: float
    MGII_REW: float
    CIV_Peak_Wavelength: float
    CIV_50_Percent_Flux_Centroid_Wavelength: float
    CIV_Flux: float
    CIV_Logarithm_of_Line_Luminosity: float
    CIV_FWHM: float
    CIV_REW: float
    LOGMBH_HB_ERR: float
    LOGMBH_MGII_ERR: float
    LOGMBH_CIV_ERR: float
    LOGMBH_ERR: float
    GAIA_G_FLUX_SNR: float
    GAIA_BP_FLUX_SNR: float


@app.get("/", response_class=HTMLResponse)
def get_ui():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Prediction Form</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                display: flex;
                justify-content: center;
                align-items: center;
                min-height: 100vh;
                margin: 0;
                background-color: #f5f5f5;
            }
            h1 { 
              text-align: center;
              margin: 20px 0;
            }

            #container {
                width: 100%;
                max-width: 600px;
                background: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
                box-sizing: border-box;
            }
            input, button {
                width: 100%;
                padding: 10px;
                margin-bottom: 15px;
                border: 1px solid #ddd;
                border-radius: 4px;
                box-sizing: border-box;
            }
            button {
                background-color: #007bff;
                color: white;
                cursor: pointer;
            }
            button:hover {
                background-color: #0056b3;
            }
            #output {
                padding: 15px;
                background-color: #f9f9f9;
                border: 1px solid #ddd;
                border-radius: 4px;
                min-height: 50px;
            }
        </style>
    </head>
    <body>
        <div id="container">
            <h1>Black Hole Mass Predictor</h1>
            <form id="predict-form">
                <input type="number" step="any" name="Z_DR16Q" placeholder="Z_DR16Q" required>
                <input type="number" step="any" name="LOGMBH_HB" placeholder="LOGMBH_HB" required>
                <input type="number" step="any" name="LOGMBH_MGII" placeholder="LOGMBH_MGII" required>
                <input type="number" step="any" name="LOGMBH_CIV" placeholder="LOGMBH_CIV" required>
                <input type="number" step="any" name="LOGMBH" placeholder="LOGMBH" required>
                <input type="number" step="any" name="LOGL5100" placeholder="LOGL5100" required>
                <input type="number" step="any" name="LOGL3000" placeholder="LOGL3000" required>
                <input type="number" step="any" name="LOGL1350" placeholder="LOGL1350" required>
                <input type="number" step="any" name="HBETA_Peak_Wavelength" placeholder="HBETA_Peak_Wavelength" required>
                <input type="number" step="any" name="HBETA_50_Percent_Flux_Centroid_Wavelength" placeholder="HBETA_50_Percent_Flux_Centroid_Wavelength" required>
                <input type="number" step="any" name="HBETA_Flux" placeholder="HBETA_Flux" required>
                <input type="number" step="any" name="HBETA_Logarithm_of_Line_Luminosity" placeholder="HBETA_Logarithm_of_Line_Luminosity" required>
                <input type="number" step="any" name="HBETA_FWHM" placeholder="HBETA_FWHM" required>
                <input type="number" step="any" name="HBETA_REW" placeholder="HBETA_REW" required>
                <input type="number" step="any" name="MGII_Peak_Wavelength" placeholder="MGII_Peak_Wavelength" required>
                <input type="number" step="any" name="MGII_50_Percent_Flux_Centroid_Wavelength" placeholder="MGII_50_Percent_Flux_Centroid_Wavelength" required>
                <input type="number" step="any" name="MGII_Flux" placeholder="MGII_Flux" required>
                <input type="number" step="any" name="MGII_Logarithm_of_Line_Luminosity" placeholder="MGII_Logarithm_of_Line_Luminosity" required>
                <input type="number" step="any" name="MGII_FWHM" placeholder="MGII_FWHM" required>
                <input type="number" step="any" name="MGII_REW" placeholder="MGII_REW" required>
                <input type="number" step="any" name="CIV_Peak_Wavelength" placeholder="CIV_Peak_Wavelength" required>
                <input type="number" step="any" name="CIV_50_Percent_Flux_Centroid_Wavelength" placeholder="CIV_50_Percent_Flux_Centroid_Wavelength" required>
                <input type="number" step="any" name="CIV_Flux" placeholder="CIV_Flux" required>
                <input type="number" step="any" name="CIV_Logarithm_of_Line_Luminosity" placeholder="CIV_Logarithm_of_Line_Luminosity" required>
                <input type="number" step="any" name="CIV_FWHM" placeholder="CIV_FWHM" required>
                <input type="number" step="any" name="CIV_REW" placeholder="CIV_REW" required>
                <input type="number" step="any" name="LOGMBH_HB_ERR" placeholder="LOGMBH_HB_ERR" required>
                <input type="number" step="any" name="LOGMBH_MGII_ERR" placeholder="LOGMBH_MGII_ERR" required>
                <input type="number" step="any" name="LOGMBH_CIV_ERR" placeholder="LOGMBH_CIV_ERR" required>
                <input type="number" step="any" name="LOGMBH_ERR" placeholder="LOGMBH_ERR" required>
                <input type="number" step="any" name="GAIA_G_FLUX_SNR" placeholder="GAIA_G_FLUX_SNR" required>
                <input type="number" step="any" name="GAIA_BP_FLUX_SNR" placeholder="GAIA_BP_FLUX_SNR" required>
                <button type="submit">Submit</button>
            </form>
            <div id="output"></div>
        </div>

        <script>
            document.getElementById("predict-form").addEventListener("submit", async function(event) {
                event.preventDefault();
                const formData = new FormData(this);
                const data = Object.fromEntries(formData.entries());
                const outputDiv = document.getElementById("output");
                outputDiv.textContent = "Loading...";

                const response = await fetch("/predict", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(data)
                });

                const result = await response.json();
                outputDiv.textContent = `Mass: ${result.prediction}`;
            });
        </script>
    </body>
    </html>
    """
    return html_content


@app.post("/predict")
def predict(data: PredictionRequest):
    input_data = np.array([
        data.Z_DR16Q, data.LOGMBH_HB, data.LOGMBH_MGII, data.LOGMBH_CIV, data.LOGMBH,
        data.LOGL5100, data.LOGL3000, data.LOGL1350, data.HBETA_Peak_Wavelength, data.HBETA_50_Percent_Flux_Centroid_Wavelength,
        data.HBETA_Flux, data.HBETA_Logarithm_of_Line_Luminosity, data.HBETA_FWHM, data.HBETA_REW,
        data.MGII_Peak_Wavelength, data.MGII_50_Percent_Flux_Centroid_Wavelength, data.MGII_Flux, data.MGII_Logarithm_of_Line_Luminosity,
        data.MGII_FWHM, data.MGII_REW, data.CIV_Peak_Wavelength, data.CIV_50_Percent_Flux_Centroid_Wavelength, data.CIV_Flux,
        data.CIV_Logarithm_of_Line_Luminosity, data.CIV_Flux, data.CIV_REW, data.LOGMBH_HB_ERR, data.LOGMBH_MGII_ERR, data.LOGMBH_CIV_ERR,
        data.LOGMBH_ERR, data.GAIA_G_FLUX_SNR, data.GAIA_BP_FLUX_SNR]).reshape(1, -1)

    prediction = model.predict(input_data)

    return {"prediction": prediction[0]}
