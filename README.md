# Factory Reallocation & Shipping Optimization

A machine-learning-based decision intelligence system for **Nassau Candy Distributor**.  
The application predicts shipping lead time, simulates alternative factory assignments, and recommends factory-product configurations that balance shipping efficiency and profitability.

## Main Features

- Executive shipping and profitability dashboard
- Product, region, and ship-mode filters
- Linear Regression, Random Forest, and Gradient Boosting comparison
- Factory reassignment scenario simulation
- Lead-time reduction, profit impact, risk, and confidence scores
- Speed-versus-profit optimization slider
- Downloadable recommendation results

## Project Structure

```text
nassau-factory-optimization/
├── app.py
├── factory_config.json
├── requirements.txt
├── README.md
├── .gitignore
└── data/
    └── nassau_candy_distributor.csv
```

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local address shown by Streamlit.

## Deploy on Streamlit Community Cloud

1. Upload all files to a public GitHub repository.
2. Sign in to Streamlit Community Cloud with GitHub.
3. Click **Create app**.
4. Select the repository and branch.
5. Set the main file path to `app.py`.
6. Click **Deploy**.

## Analytical Methodology

1. Parse order and shipping dates.
2. Calculate shipping lead time.
3. Map products to their current factories.
4. Estimate factory-to-region distance using factory coordinates and regional centroids.
5. Train and compare three regression models.
6. Simulate every factory option for the selected product.
7. rank scenarios using lead-time improvement, profit stability, and risk.

## Important Limitation

The source dataset contains historical product assignments but does not provide actual freight charges, customer latitude/longitude, factory capacity, or previous factory-reallocation experiments. Therefore, distance, logistics cost, profit impact, risk, and alternate-factory outcomes are analytical estimates for decision support, not guaranteed operational results.

## Author

Created as an academic analytics and machine-learning project.
