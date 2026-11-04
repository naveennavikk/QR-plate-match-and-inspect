# QR Plate Match and Inspect API

This repository contains the FastAPI service, a small HTML/CSS/JavaScript test page, the image matcher, and trained inspector weights. Your team's backend can call the same API endpoints. The service compares the reference QR plate and its surrounding background with the property image. If they match, the inspector analyzes the reference plate image.

## Run the API

Install the dependencies in your Python environment:

```powershell
python -m pip install -r requirements.txt
```

From this folder, start the service:

```powershell
uvicorn api:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` to test one or multiple image pairs in the browser. The interactive API schema is at `http://localhost:8000/docs`; health check: `http://localhost:8000/health`.

## Backend request

Send `POST http://localhost:8000/api/inspect` with JSON containing the two required images. Each image can be raw base64 or a `data:image/...;base64,...` string. Optional `houseId` and `userId` values are echoed back for record correlation. Other fields in the existing property record may also be included and are ignored by the image analysis.

```json
{
  "houseId": 7044,
  "QRCodeImage": "<base64 reference plate image>",
  "HouseImage": "<base64 property image>"
}
```

For multiple pairs in one request, send `POST http://localhost:8000/api/inspect/batch` with an `items` array. Each item uses the same record shape. A batch can contain up to 20 pairs; results are returned in the same order and include the zero-based `index` plus any supplied IDs. A bad image in one item is reported for that item and does not stop the other pairs from being processed.

```json
{
  "items": [
    {
      "houseId": 7044,
      "QRCodeImage": "<base64 reference plate image 1>",
      "HouseImage": "<base64 property image 1>"
    },
    {
      "houseId": 7045,
      "QRCodeImage": "<base64 reference plate image 2>",
      "HouseImage": "<base64 property image 2>"
    }
  ]
}
```

## Response

`QRStatus` follows the existing application's convention:

- `1`: approve; `Remark` is empty.
- `0`: reject; `Remark` explains the rejection.
- `null`: manual review; do not update the existing record. The existing `Remark` is returned unchanged when supplied.

Example automatic rejection:

```json
{
  "QRStatus": 0,
  "Remark": "Reference plate or its surrounding background does not match the property image.",
  "manual_review": false,
  "houseId": 7044
}
```

The calling backend should persist `QRStatus` and `Remark` only for automatic results and leave the record unchanged when `QRStatus` is `null`. It can set `QRStatusDate` and `modified` according to its own record-update rules.
