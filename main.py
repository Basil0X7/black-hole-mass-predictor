from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import numpy as np
import joblib

app = FastAPI()
model = joblib.load("models/random_forest_regression.pkl")

class PredictionRequest(BaseModel):
    z_dr16q: float
    logmbh_hb: float
    logmbh_mgii: float
    logmbh_civ: float
    logl5100: float
    logl3000: float
    logl1350: float
    hbeta: float
    mgii: float
    civ: float
    logmbh_hb_err: float
    logmbh_mgii_err: float
    logmbh_civ_err: float
    logmbh_err: float
    gaia_g_flux_snr: float
    gaia_bp_flux_snr: float


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
                <input type="number" step="any" name="z_dr16q" placeholder="z_dr16q" required>
                <input type="number" step="any" name="logmbh_hb" placeholder="logmbh_hb" required>
                <input type="number" step="any" name="logmbh_mgii" placeholder="logmbh_mgii" required>
                <input type="number" step="any" name="logmbh_civ" placeholder="logmbh_civ" required>
                <input type="number" step="any" name="logl5100" placeholder="logl5100" required>
                <input type="number" step="any" name="logl3000" placeholder="logl3000" required>
                <input type="number" step="any" name="logl1350" placeholder="logl1350" required>
                <input type="number" step="any" name="hbeta" placeholder="hbeta" required>
                <input type="number" step="any" name="mgii" placeholder="mgii" required>
                <input type="number" step="any" name="civ" placeholder="civ" required>
                <input type="number" step="any" name="logmbh_hb_err" placeholder="logmbh_hb_err" required>
                <input type="number" step="any" name="logmbh_mgii_err" placeholder="logmbh_mgii_err" required>
                <input type="number" step="any" name="logmbh_civ_err" placeholder="logmbh_civ_err" required>
                <input type="number" step="any" name="logmbh_err" placeholder="logmbh_err" required>
                <input type="number" step="any" name="gaia_g_flux_snr" placeholder="gaia_g_flux_snr" required>
                <input type="number" step="any" name="gaia_bp_flux_snr" placeholder="gaia_bp_flux_snr" required>
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
    input_data = np.array([data.z_dr16q, data.logmbh_hb, data.logmbh_mgii, data.logmbh_civ, data.logl5100,
                           data.logl3000, data.logl1350, data.hbeta, data.mgii, data.civ,
                           data.logmbh_hb_err, data.logmbh_mgii_err, data.logmbh_civ_err,
                           data.logmbh_err, data.gaia_g_flux_snr, data.gaia_bp_flux_snr]).reshape(1, -1)

    prediction = model.predict(input_data)

    return {"prediction": prediction[0]}
