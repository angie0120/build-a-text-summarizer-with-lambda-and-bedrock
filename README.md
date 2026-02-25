# Build a Text Summarizer with AWS Lambda + Amazon Bedrock (Beginner)

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
Make sure the model you choose is available in your region.

```python
AWS_REGION_BEDROCK = "us-east-1"
MODEL_ID = "amazon.nova-micro-v1:0"
```

#### Step 2 - Read the Input

Inside lambda_handler, the first thing we do is grab the values from the event:

```python
text = event.get("text")
points = event.get("points")
```

When we click “Test” in Lambda, AWS sends our JSON into this function as a dictionary called event.
So here, we’re just pulling out:
- The paragraph to summarize
- The number of bullet points

#### Step 3 - Build the Prompt

Next, we create the message we want to send to Nova.

```python
prompt = f"""Text:
{text}

Summarize the text into {points} bullet points.
"""
```

Now here’s something important. Nova doesn’t automatically know we want a summary. There’s no special summary mode.
We tell it what to do right here:
This sentence is the instruction - “Summarize the text into 3 bullet points”,
If I changed that line to “Rewrite the paragraph as a poem,” I’d get a poem instead.

Then we wrap that prompt in Nova’s required format:

```python
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
```

Nova expects a “messages” structure similar to chatting with an AI.

#### Step 4 - Call Amazon Bedrock

Now that we’ve built our instruction, we send it to Nova:

```python
response = client.invoke_model(
modelId=MODEL_ID,
body=json.dumps(request_body),
accept="application/json",
contentType="application/json",
)
```

This sends our prompt to Nova. Nova processes it and sends back a response.

#### Step 5 - Extract the Summary

Finally, we extract the summary text:

```python
summary = response_body["output"]["message"]["content"][0]["text"]
```

And return it as a JSON response:

```python
return _response(200, {"summary": summary.strip()})
```

---

## Example Test Event

```josn
{
  "text": "Dogs and cats are two of the most popular pets in the world, each offering different types of companionship. Dogs are known for their loyalty and energetic nature, making them great companions for active individuals and families. Cats are more independent and typically require less daily attention, which appeals to busy people or those living in smaller spaces. Both pets can improve mental well-being by reducing stress and loneliness, but the right choice ultimately depends on a person’s lifestyle and preferences.",
  "points": 3
}
```

### Example Response

```
{
  "statusCode": 200,

  },
  "body": "{\"summary\": \"- Dogs are loyal and energetic, ideal for active individuals and families.\\n- Cats are independent and require less daily attention, suitable for busy people or small spaces.\\n- Both dogs and cats can enhance mental well-being by reducing stress and loneliness, with the best choice depending on personal lifestyle and preferences.\"}"
}

```

---

## Required Permissions

Your Lambda execution role must allow: ```AmazonBedrockFullAccess```

---

## Next Steps

From here, you could:
- Connect it to a website
- Process documents automatically
- Extend it into a larger AI workflow

---

1. YouTube tutorial (link)
2. [Lambda Setup Guide](https://github.com/angie0120/lambda-text-summarizer)

---
