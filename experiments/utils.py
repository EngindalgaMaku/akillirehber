"""Ortak yardımcı fonksiyonlar."""

import json
import os
import sys
import requests
from datetime import datetime


def load_config(config_path="experiment_config.json"):
    """Deney konfigürasyonunu yükle."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_dir, config_path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_auth_token(base_url, email, password):
    """API'den JWT token al."""
    resp = requests.post(
        f"{base_url}/api/auth/login",
        data={"username": email, "password": password},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def auth_headers(token):
    """Authorization header döndür."""
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
