import os
import json
from fastapi import FastAPI, Request, HTTPException
import uvicorn
import requests
from dotenv import load_dotenv
import pandas as pd
from mlflow.models import Model
import mlflow
from collections import Counter
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np
import shap



class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


#load_dotenv()
app = FastAPI()
model = mlflow.sklearn.load_model(os.environ.get("MODEL_PATH"))
client_data = joblib.load(os.environ.get("DATA_PATH"))
explainer = shap.Explainer(model.named_steps['lgbmclassifier'])

def scale_data(data):
    for column in client_data.columns:
        data[column] = data[column].astype(client_data[column].dtype)

    cont = data.select_dtypes(exclude='int8')
    binn = data.select_dtypes(include='int8')
    scaler = StandardScaler()
    scaler.fit(cont.drop(labels='SK_ID_CURR', axis=1))
    scaled_cont = scaler.transform(cont.drop(labels='SK_ID_CURR', axis=1))
    scaled_cont = pd.DataFrame(scaled_cont, columns=cont.drop(labels='SK_ID_CURR', axis=1).columns)
    res = pd.concat([scaled_cont, binn], axis=1)
    res['SK_ID_CURR'] = data['SK_ID_CURR']
    return res

client_data_scaled = scale_data(client_data)


@app.post("/search")
async def search_client(request : Request):
    data = await request.json()
    sk_id = data.get('sk_id', None)
    if not sk_id:
        raise HTTPException(400, detail="Not a valid SK_ID")
    hit = client_data.loc[client_data['SK_ID_CURR'] == sk_id].reset_index(drop=True).to_json()
    if not hit:
        raise HTTPException(404, detail="Client not found!")
    return hit


@app.post("/infer")
async def infer(request : Request):
    data = await request.json()
    df = pd.DataFrame(data)
    df['SK_ID_CURR'] = df['SK_ID_CURR'].astype('int64')
    sk_id = df['SK_ID_CURR'].tolist()[0]
    df_index = client_data.loc[client_data["SK_ID_CURR"] == sk_id].index[0]
    for column in client_data.columns:
        client_data.at[df_index, column] = df.at[df.index[0], column]

    print("Data should change here: ")
    print(client_data.loc[client_data['SK_ID_CURR'] == sk_id])
    client_data_scaled = scale_data(client_data)
    to_pred = client_data_scaled.loc[client_data_scaled['SK_ID_CURR'] == sk_id]
    to_pred = to_pred.drop(labels='SK_ID_CURR', axis=1)
    answer = model.predict(to_pred)[0]
    answer_prob = model.predict_proba(to_pred)[:, 1].tolist()[0]
    res_obj = {"answer": answer, "answer_probability": answer_prob}
    print(res_obj["answer"])
    return json.dumps(res_obj, cls=NpEncoder)


@app.post("/explain")
async def explain(request : Request):
    data = await request.json()
    df = pd.DataFrame(data)
    df['SK_ID_CURR'] = df['SK_ID_CURR'].astype('int64')
    sk_id = df['SK_ID_CURR'].tolist()[0]
    to_pred = client_data_scaled.loc[client_data_scaled['SK_ID_CURR'] == sk_id]
    to_pred = to_pred.drop(labels='SK_ID_CURR', axis=1)
    shap_values = explainer.shap_values(to_pred)
    res_obj = {"shap_values": shap_values[0][0]}
    return json.dumps(res_obj, cls=NpEncoder)


@app.post("/compare")
async def send_all(request : Request):
    to_pred = client_data_scaled.drop(labels='SK_ID_CURR', axis=1)
    targets = model.predict(to_pred)
    temp = client_data
    temp['targets'] = targets
    temp = temp[temp['targets'] == 0].drop(labels='targets', axis=1)
    client_data.drop(labels='targets', axis=1, errors="ignore", inplace=True)
    return temp.to_json(orient='records')
    
if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=int(os.getenv('APP_PORT', '4545')))
