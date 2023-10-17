import pytest
import requests
import json
import os

api_url = os.environ.get("API_URL")

def test_search_ok():
    payload = json.dumps({"sk_id": 100005})
    response = requests.post(url=api_url, headers={"Content-type": "application/json"}, data=payload)
    assert response.status_code == 200

def test_search_ok_not_found():
    payload = json.dumps({"sk_id": -42})
    response = requests.post(url=api_url, headers={"Content-type": "application/json"}, data=payload)
    assert response.status_code == 200

def test_search_bad():
    payload = json.dumps({})
    response = requests.post(url=api_url, headers={"Content-type": "application/json"}, data=payload)
    assert response.status_code == 400
