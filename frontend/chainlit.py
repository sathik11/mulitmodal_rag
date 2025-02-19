import chainlit as cl
import base64
import requests

# Example placeholder function to call OpenAI chat completion.
# In a real app, you'd replace this with actual API calls.
def call_openai_api(user_input, conversation, image_base64=None):
    response_text = "Hi this is my response"
    image = requests.get("https://www.shutterstock.com/image-vector/budget-2025-logo-design-banner-260nw-2229933743.jpg")
    response_image_base64 = base64.b64decode(image.content)
    citations = [
        "https://immigration.gov.ph/wp-content/uploads/2024/11/2025-ANNUAL-REPORT-GUIDELINES-1.pdf.pdf"
    ]
    return response_text, response_image_base64, citations

@cl.on_chat_start
async def start():
    # content="# RAG Chat Interface\n![Logo](https://placehold.co/200x60)"
    await cl.Message(
        content="RAG Chat Interface"
    ).send()

@cl.on_message
async def main(message: str):
    # Use an attribute on cl.session to store conversation history
    if not hasattr(cl.session, "conversation"):
        cl.session.conversation = []
    
    conversation = cl.session.conversation

    # Capture user text input
    conversation.append({"role": "user", "content": message})

    # Call backend (OpenAI or custom RAG service)
    ai_text, ai_image_b64, citations = call_openai_api(message, conversation)

    # Build AI response
    if ai_image_b64:
        image = "https://www.shutterstock.com/image-vector/budget-2025-logo-design-banner-260nw-2229933743.jpg"
        # response_image_base64 = base64.b64decode(image.content)
        # ai_image = response_image_base64 #base64.b64decode(ai_image_b64)
        cl.Image(path=image, name="AI Image Response", display="inline")
        await cl.Message(
            content=ai_text,
            elements=[cl.Image(url=image, name="AI Image Response", display="inline")]
        ).send()
    else:
        await cl.Message(content=ai_text).send()

    # Display citations if present
    if citations:
        citation_text = "**Latest Citations:**\n" + "\n".join([f"[View Document]({c})" for c in citations])
        await cl.Message(content=citation_text).send()

    conversation.append({"role": "assistant", "content": ai_text})