import json
import boto3

AWS_REGION_BEDROCK = "us-east-1"
MODEL_ID = "amazon.nova-micro-v1:0"

client = boto3.client("bedrock-runtime", region_name=AWS_REGION_BEDROCK)

def lambda_handler(event, context):

    # Read input from test event
    text = event.get("text")
    points = event.get("points")

    if not text or points is None:
        return _response(400, {"error": "text and points required"})

    try:
        points = int(points)
    except ValueError:
        return _response(400, {"error": "points must be an integer"})

    # Build request body for Nova
    prompt = f"""Text:
{text}

Summarize the text into {points} numbered points.

Summarize the text into {points} numbered points.

Rules:
- Use numbers starting from 1
- One sentence per point
- Keep each point concise
"""

    request_body = {
        "schemaVersion": "messages-v1",
        "messages": [
            {
                "role": "user",
                "content": [{"text": prompt}]
            }
        ],
        "inferenceConfig": {
            "maxTokens": 512,
            "temperature": 0,
            "topP": 1
        }
    }

    # Call Bedrock
    response = client.invoke_model(
        modelId=MODEL_ID,
        body=json.dumps(request_body),
        accept="application/json",
        contentType="application/json",
    )

    response_body = json.loads(response["body"].read())

    # Extract summary text
    summary = response_body["output"]["message"]["content"][0]["text"]

    return _response(200, {"summary": summary.strip()})


def _response(status_code, body):
    return {
        "statusCode": status_code,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }