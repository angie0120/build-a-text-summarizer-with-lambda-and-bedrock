# Build a Text Summarizer with AWS Lambda + Amazon Bedrock (Beginner)

In this project, you'll build a simple text summarizer using:

- **AWS Lambda**
- **Amazon Bedrock** (Amazon Nova Micro model)
- **Python**

This project matches my [YouTube video](https://youtu.be/3_t5cpJpLi4)

---

## What This Project Builds

Let’s say you have a paragraph like this:

> Dogs and cats are two of the most popular pets in the world, each offering different types of companionship. Dogs are known for their loyalty and energetic nature, making them great companions for active individuals and families. Cats are more independent and typically require less daily attention, which appeals to busy people or those living in smaller spaces. Both pets can improve mental well-being by reducing stress and loneliness, but the right choice ultimately depends on a person’s lifestyle and preferences.

Instead of manually shrinking this into bullet points, this project builds a Lambda function that:

1. Reads your input  
2. Builds a prompt  
3. Calls Amazon Bedrock  
4. Returns a structured summary  

---

## Lambda Setup (Important)

Instead of running the Python scripts locally, this project runs the code in the cloud using AWS Lambda.

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
```

---

## How the Code Works (Step-by-Step)

### Step 1 - Set Region and Model
Make sure the model you choose is available in your region.

```python
AWS_REGION_BEDROCK = "us-east-1"
MODEL_ID = "amazon.nova-micro-v1:0"
```

### Step 2 - Read the Input

Inside lambda_handler, the function first retrieves the values from the event:

```python
text = event.get("text")
points = event.get("points")
```

When we click “Test” in Lambda, AWS sends our JSON into this function as a dictionary called event.
So here, we’re just pulling out:
- The paragraph to summarize
- The number of bullet points

### Step 3 - Build the Prompt

Next, we create the message we want to send to Nova.

```python
prompt = f"""Text:
{text}

Summarize the text into {points} numbered points.

Rules:
- Use numbers starting from 1
- One sentence per point
- Keep each point concise
"""
```

Here’s something important to note - Nova doesn’t automatically know we want a summary. There’s no special summary mode.
We have to give it instructions. This sentence in the Python script is the instruction: “Summarize the text into {points} numbered points.”
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

### Step 4 - Call Amazon Bedrock

After building the instruction, the request is sent to Nova using the Bedrock Runtime API:

```python
response = client.invoke_model(
modelId=MODEL_ID,
body=json.dumps(request_body),
accept="application/json",
contentType="application/json",
)
```

**This is the actual API call.**

The `invoke_model()` function sends a request from AWS Lambda to the Amazon Bedrock service. Behind the scenes, this makes a network request to AWS, which forwards the prompt to the Nova model and waits for a response.

In simple terms: **Lambda → Bedrock API → Nova model → Response → Lambda**

> Everything before the `invoke_model()` call prepares the request.
> 
> The `invoke_model()` call sends the request to Amazon Bedrock.
> 
> Everything after it processes the response.

### Step 5 - Extract the Summary

Finally, the summary text is extracted:

```python
summary = response_body["output"]["message"]["content"][0]["text"]
```

And return it as a JSON response:

```python
return _response(200, {"summary": summary.strip()})
```

---

## Example Test Event

```
{
  "text": "Dogs and cats are two of the most popular pets in the world, each offering different types of companionship. Dogs are known for their loyalty and energetic nature, making them great companions for active individuals and families. Cats are more independent and typically require less daily attention, which appeals to busy people or those living in smaller spaces. Both pets can improve mental well-being by reducing stress and loneliness, but the right choice ultimately depends on a person’s lifestyle and preferences.",
  "points": 3
}
```

### Example Response

```
{
  "statusCode": 200,
  "headers": {
    "Content-Type": "application/json"
  },
  "body": "{\"summary\": \"1. Dogs are loyal and energetic, ideal for active individuals and families.  \\n2. Cats are independent and require less daily attention, suitable for busy people or small spaces.  \\n3. Both pets can enhance mental well-being by reducing stress and loneliness.\"}"
}

```

---

## Required Permissions

Lambda execution role used in the demo: ```AmazonBedrockFullAccess```

---

## Next Steps

From here, you could:
- Connect it to a website
- Process documents automatically

---

## GRC Use Case

This isn’t just a summarizer, it’s an evidence helper. In GRC, you often need to turn long text (policies, vendor questionnaires, incident summaries) into short, consistent notes for audits. This Lambda function standardizes that output into numbered points you can store as supporting evidence (with human review). It aligns with documentation and review expectations across SOC 2, NIST 800-53, and ISO/IEC 42001.

---

1. [YouTube](https://youtu.be/3_t5cpJpLi4) Walkthrough
2. [Lambda Setup Guide](https://github.com/angie0120/lambda-text-summarizer)

---
