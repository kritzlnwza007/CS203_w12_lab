
import os
import requests
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("EXCHANGE_RATE_API_KEY")
BASE_URL = f"https://v6.exchangerate-api.com/v6/{self.api_key}/latest/"

def convert_currency(amount: float, from_currency: str, to_currency: str) -> Dict[str, Any]:
    """
    Convert an amount from one currency to another using the ExchangeRate API.
    Returns a dictionary containing the rate and the converted value.
    """
    if not API_KEY:
        return {"error": "API key missing."}

    url = f"{BASE_URL}{from_currency.upper()}"
    response = requests.get(url)
    data = response.json()

    if data.get("result") != "success":
        return {"error": f"API error: {data.get('error-type', 'unknown error')}"}

    rates = data.get("conversion_rates", {})
    if to_currency.upper() not in rates:
        return {"error": f"No rate found for {to_currency}"}

    rate = rates[to_currency.upper()]
    converted_amount = amount * rate

    return {
        "from_currency": from_currency.upper(),
        "to_currency": to_currency.upper(),
        "amount": amount,
        "rate": rate,
        "converted": converted_amount
    }
