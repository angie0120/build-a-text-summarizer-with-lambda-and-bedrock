# Build a Text Summarizer with AWS Lambda + Amazon Bedrock (Beginners)

In this project, you'll build a simple text summarizer using:

- **AWS Lambda**
- **Amazon Bedrock**
- **Amazon Nova Micro**
- **Python**

This project matches my YouTube tutorial. Watch the full video here: https://youtube.com/YOUR_VIDEO_LINK_HERE

---

## What We're Building

Let’s say you have a paragraph like this:

> Pets like cats and dogs are wonderful companions because they bring comfort, joy, and love into our lives. They can help reduce stress, ease loneliness, and even encourage people to be more active through play and walks. Caring for a pet also fosters a sense of responsibility and routine, which can be especially helpful for children or anyone needing stability. Beyond the emotional benefits, pets create strong bonds that often feel like true friendship, always offering affection and loyalty in return for a little care and attention.

Instead of manually shrinking this into bullet points, we’ll build a Lambda function that:

1. Reads your input  
2. Builds a prompt  
3. Calls Amazon Bedrock  
4. Returns a structured summary  

That’s it.

---

## Architecture Overview

Lambda (Python)  
→ Amazon Bedrock (Nova Micro)  
→ JSON Summary Response  

---

## Lambda Setup (Important)

In previous projects, we ran Python scripts locally.  
In this project, we're running our code in the cloud using AWS Lambda.

To follow along, you’ll need to create the Lambda function first.

I’ve included step-by-step instructions (with screenshots) here: [Lambda Setup Guide](https://github.com/angie0120/lambda-text-summarizer)

That guide covers:
- Creating the Lambda function  
- Choosing the Python runtime  
- Attaching Bedrock permissions  
- Creating the test event  

---

## The Code

Below is the full `lambda_function.py` used in the tutorial.

```python
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

    request_body = build_nova_request(text, points)

    response = client.invoke_model(
        modelId=MODEL_ID,
        body=request_body,
        accept="application/json",
        contentType="application/json",
    )

    response_body = json.loads(response["body"].read())
    summary = response_body["output"]["message"]["content"][0]["text"]

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
```

---

## How the Code Works (Step-by-Step)

#### Step 1 - Set Region and Model

```python
AWS_REGION_BEDROCK = "us-east-1"
MODEL_ID = "amazon.nova-micro-v1:0"
```

We specify:

- The AWS region
- The Nova model we want to use
Make sure the model you choose is available in your region.

#### Step 2 - Read the Input

When you test the Lambda function, you send a JSON object like this:

```json
{
  "text": "Your paragraph here...",
  "points": 3
}
```

Lambda automatically converts that into a Python dictionary called event.

We extract the values we need:

```python
text = event.get("text")
points = event.get("points")
```

#### Step 3 - Build the Prompt

Nova uses a messages format, similar to chatting with an AI.

We build a prompt and send it inside:

```python
"schemaVersion": "messages-v1",
"messages": [...]
"inferenceConfig": {...}
```

The inference settings control:
- Maximum length
- Creativity
- Consistency

#### Step 4 - Call Amazon Bedrock

```python
response = client.invoke_model(...)
```

This sends the prompt to Nova and waits for the summary response.

#### Step 5 - Extract the Summary

Nova returns structured content blocks.

We extract the text from:

```python
response_body["output"]["message"]["content"][0]["text"]
```

That’s the summary.

---

#### Example Test Event

```josn
{
  "text": "Pets like cats and dogs are wonderful companions because they bring comfort, joy, and love into our lives. They can help reduce stress, ease loneliness, and even encourage people to be more active through play and walks. Caring for a pet also fosters a sense of responsibility and routine, which can be especially helpful for children or anyone needing stability. Beyond the emotional benefits, pets create strong bonds that often feel like true friendship, always offering affection and loyalty in return for a little care and attention.",
  "points": 3
}
```

#### Example Response

```json
{
  "statusCode": 200,
  "body": "{\"summary\":\"- Pets like cats and dogs bring comfort, joy, and love into our lives.\\n- They help reduce stress, ease loneliness, and encourage physical activity.\\n- Caring for a pet fosters a sense of responsibility and routine.\"}"
}
```

---

### Required Permissions

Your Lambda execution role must allow:

```bedrock:InvokeModel```

For beginners, attaching:

```AmazonBedrockFullAccess```

is the simplest option.

---

### Next Steps

From here, you could:
- Turn this into an API
- Connect it to a website
- Process documents automatically
- Extend it into a larger AI workflow

---

📺 YouTube tutorial:
https://youtube.com/YOUR_VIDEO_LINK_HERE

📘 Lambda setup guide:
https://github.com/YOUR_USERNAME/YOUR_REPO/blob/main/LAMBDA_SETUP.md
