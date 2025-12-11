# Milk Tea & Dessert Location Intelligence – Chicago

Streamlit app that recommends new bubble tea / dessert shop locations using:

[Streamlit]https://locationrecommendation-9kcuj7aatzb3hvotuqt54b.streamlit.app/

- Community demographics (ACS)
- CTA & Divvy accessibility
- Crime data
- Yelp competitors
- Custom scoring & white-space analysis

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Data

Place the following CSV files under data/:

```
final_master_clean.csv
yelp_with_comm.csv
lincoln_park_recommendations.csv
```
[Yelp API]https://docs.developer.yelp.com/reference/v2_ai_chat
