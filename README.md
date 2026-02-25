# Build a Text Summarizer with AWS Lambda + Amazon Bedrock (Beginners)

In this project, you'll build a simple text summarizer using:

- **AWS Lambda**
- **Amazon Bedrock** (Amazon Nova Micro model)
- **Python**

This project matches my YouTube tutorial. Watch the full video here: https://youtube.com/YOUR_VIDEO_LINK_HERE

---

## What We're Building

Let’s say you have a paragraph like this:

> Dogs and cats are two of the most popular pets in the world, each offering different types of companionship. Dogs are known for their loyalty and energetic nature, making them great companions for active individuals and families. Cats are more independent and typically require less daily attention, which appeals to busy people or those living in smaller spaces. Both pets can improve mental well-being by reducing stress and loneliness, but the right choice ultimately depends on a person’s lifestyle and preferences.

Instead of manually shrinking this into bullet points, we’ll build a Lambda function that:

1. Reads your input  
2. Builds a prompt  
3. Calls Amazon Bedrock  
4. Returns a structured summary  

---

## Lambda Setup (Important)

In previous projects, we ran Python scripts locally. In this project, we're running our code in the cloud using AWS Lambda.

To follow along, you’ll need to create the Lambda function first. I’ve included step-by-step instructions (with screenshots) here: [Lambda Setup Guide](https://github.com/angie0120/lambda-text-summarizer)

That guide covers:
- Creating the Lambda function  
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

Summarize the text into {points} bullet points.
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
  "text": "Dogs and cats are two of the most popular pets in the world, each offering different types of companionship. Dogs are known for their loyalty and energetic nature, making them great companions for active individuals and families. Cats are more independent and typically require less daily attention, which appeals to busy people or those living in smaller spaces. Both pets can improve mental well-being by reducing stress and loneliness, but the right choice ultimately depends on a person’s lifestyle and preferences.",
  "points": 3
}
```

#### Example Response

```json
{
  "statusCode": 200,

  },
  "body": "{\"summary\": \"- Dogs are loyal and energetic, ideal for active individuals and families.\\n- Cats are independent and require less daily attention, suitable for busy people or small spaces.\\n- Both dogs and cats can enhance mental well-being by reducing stress and loneliness, with the best choice depending on personal lifestyle and preferences.\"}"
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

[Lambda Setup Guide](https://github.com/angie0120/lambda-text-summarizer)
