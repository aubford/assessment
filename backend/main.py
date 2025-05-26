from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import openai
import os
import json
import itertools
from datetime import datetime

app = FastAPI()

order_id_generator = itertools.count(1)
all_orders: dict[int, dict] = {}


class OrderRequest(BaseModel):
    message: str


class OrderResponse(BaseModel):
    success: bool


class CreateOrderSchema(BaseModel):
    """
    This tool creates a new order. Use this when a customer wants to purchase burgers, fries, and/or drinks.
    Parse the order carefully and determine the number of each item (burgers, fries, drinks) that the user wants.
    Provide the count of burgers, fries, and drinks that comprise the order.
    """

    burgers: int = Field(ge=0, description="The number of burgers the user wants")
    fries: int = Field(ge=0, description="The number of fries the user wants")
    drinks: int = Field(ge=0, description="The number of drinks the user wants")


class CancelOrderSchema(BaseModel):
    """
    This tool cancels an order. Use this when a customer wants to cancel their order.
    Parse the order carefully and output the order ID that the user wants to cancel.
    """

    order_id: int = Field(
        gt=0, description="The order ID of the order the user wants to cancel"
    )


def create_order_tool(burgers: int = 0, fries: int = 0, drinks: int = 0) -> None:
    """Create a new order as specified by the LLM and add to all_orders"""

    if burgers + fries + drinks < 1:
        raise HTTPException(
            status_code=400, detail="Order must contain at least one item"
        )

    order_id = next(order_id_generator)

    order = {
        "id": order_id,
        "burgers": burgers,
        "fries": fries,
        "drinks": drinks,
        "timestamp": datetime.now().isoformat(),
    }

    all_orders[order_id] = order


def cancel_order_tool(order_id: int) -> dict:
    """Remove an existing order from all_orders by the order ID specified by the LLM"""

    if order_id not in all_orders:
        raise HTTPException(status_code=400, detail=f"Order #{order_id} not found")

    del all_orders[order_id]

    return {"success": True, "message": f"Order #{order_id} has been cancelled"}


system_message = """
You are a drive-thru order processing assistant.

# Task

Assist in processing drive-through orders by determining the appropriate action based on the user's request: Either creating a new order or canceling an existing order. 

The user request will always be one of:
1. Create a new order of a specified number of burgers, fries, and/or drinks.
2. Cancel an existing order for a given order ID.

# Examples

<user>
Hi, how are you today? My friend and I would each like a fries and a drink.
</user>

<assistant>
    <function_call> create_order(burgers=0, fries=2, drinks=2) </function_call>
</assistant>

<user>
Hi, I'd like to cancel my order. It's #1.
</user>

<assistant>
    <function_call> cancel_order(order_id=1) </function_call>
</assistant>

Always call exactly one of the two provided tools (create_order or cancel_order) to fulfill the user's request.
""".strip()


print(system_message)


@app.post("/process_order", response_model=OrderResponse)
async def process_order(request: OrderRequest):
    try:
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        tools = [
            openai.pydantic_function_tool(
                CreateOrderSchema,
                name="create_order",
            ),
            openai.pydantic_function_tool(
                CancelOrderSchema,
                name="cancel_order",
            ),
        ]

        response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {
                    "role": "developer",
                    "content": system_message,
                },
                {"role": "user", "content": request.message},
            ],
            tools=tools,
            tool_choice="required",
            parallel_tool_calls=False,
        )

        if not response.choices or not response.choices[0].message.tool_calls:
            raise HTTPException(status_code=400, detail="No tool call received from AI")

        tool_call = response.choices[0].message.tool_calls[0]
        function_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)

        if function_name == "create_order":
            create_order_tool(**arguments)
        elif function_name == "cancel_order":
            cancel_order_tool(**arguments)
        else:
            raise HTTPException(status_code=400, detail="Unknown function called")

        return OrderResponse(success=True)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing order: {str(e)}")


@app.get("/orders")
async def get_orders():
    return {"orders": list(all_orders.values())}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", reload=True)
