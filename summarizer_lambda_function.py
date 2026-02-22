import json
import boto3

AWS_REGION_BEDROCK = "us-east-1"
MODEL_ID = "amazon.nova-micro-v1:0"

client = boto3.client("bedrock-runtime", region_name=AWS_REGION_BEDROCK)

def lambda_handler(event, context):
    text = event.get("text")
    points = event.get("points")

    if not text or points is None:
        return _resp(400, {"error": "text and points required!"})

    try:
        points = int(points)
    except ValueError:
        return _resp(400, {"error": "points must be an integer"})

    if points < 1 or points > 10:
        return _resp(400, {"error": "points must be between 1 and 10"})

    request_body = build_nova_request(text, points)

    response = client.invoke_model(
        modelId=MODEL_ID,
        body=request_body,
        accept="application/json",
        contentType="application/json",
    )

    response_body = json.loads(response["body"].read())

    # Nova returns content blocks; text is an optional block type.
    content_blocks = response_body["output"]["message"]["content"]
    summary = next((b["text"] for b in content_blocks if "text" in b), "")

    if not summary:
        return _resp(500, {"error": "No text returned in model response", "raw": response_body})

    return _resp(200, {"summary": summary.strip()})

def build_nova_request(text: str, points: int) -> str:
    prompt = f"""Text:
    {text}

    Summarize the text into EXACTLY {points} bullet points.

    Rules:
    - Use '-' for each bullet
    - One sentence per bullet
    - Do not add any extra commentary
    """

    native_request = {
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

    return json.dumps(native_request)

def _resp(status_code: int, body: dict):
    return {
        "statusCode": status_code,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }