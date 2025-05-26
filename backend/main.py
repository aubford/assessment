from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import openai
import os
import json
import itertools
from typing import Literal
from datetime import datetime
from openai.types.responses import Response, ResponseFunctionToolCall

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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


class Order(BaseModel):
    id: int
    burgers: int
    fries: int
    drinks: int
    timestamp: str


class OrderResponse(BaseModel):
    orders: list[Order]


class OrderRequest(BaseModel):
    message: str


class CreateOrder(BaseModel):
    """
    This tool creates a new order. Use this when a customer wants to purchase burgers, fries, and/or drinks.
    Parse the order carefully and determine the number of each item (burgers, fries, drinks) that the user wants.
    Provide the count of burgers, fries, and drinks that comprise the order.
    """

    burgers: int = Field(
        ge=0, lt=500, description="The number of burgers the user wants to order"
    )
    fries: int = Field(ge=0, lt=500, description="The number of fries the user wants")
    drinks: int = Field(ge=0, lt=500, description="The number of drinks the user wants")


class CancelOrder(BaseModel):
    """
    This tool cancels an order. Use this when a customer wants to cancel their order.
    Parse the order carefully and output the order ID that the user wants to cancel.
    """

    order_id: int = Field(
        gt=0, description="The order ID of the order the user wants to cancel"
    )


order_id_generator = itertools.count(1)
all_orders: dict[int, Order] = {}


def create_order_tool(burgers: int = 0, fries: int = 0, drinks: int = 0) -> None:
    """Create a new order as specified by the LLM and add to all_orders"""

    if burgers + fries + drinks < 1:
        raise HTTPException(
            status_code=502, detail="Order must contain at least one item"
        )

    order_id = next(order_id_generator)

    order = Order(
        id=order_id,
        burgers=burgers,
        fries=fries,
        drinks=drinks,
        timestamp=datetime.now().isoformat(),
    )

    all_orders[order_id] = order


def cancel_order_tool(order_id: int) -> None:
    """Remove an existing order from all_orders by the order ID specified by the LLM"""

    if order_id not in all_orders:
        raise HTTPException(status_code=502, detail=f"Order #{order_id} not found")

    del all_orders[order_id]


def responses_api_pydantic_function_tool(model: type[BaseModel], name: str) -> dict:
    """Converter from completions API format so we can use Pydantic with new Responses API"""
    completions_api_format = openai.pydantic_function_tool(model, name=name)
    return {
        "type": "function",
        **completions_api_format["function"],
    }


tools = [
    responses_api_pydantic_function_tool(
        CreateOrder,
        name="create_order",
    ),
    responses_api_pydantic_function_tool(
        CancelOrder,
        name="cancel_order",
    ),
]


def llm_bad_response(response: Response) -> str | Literal[False]:
    """Validate the response from OpenAI"""
    if response.error:
        return response.error.message
    if len(response.output) != 1 or response.output[0].type != "function_call":
        return "Bad tool call response from LLM"
    if response.status in ["failed", "incomplete"]:
        return "LLM failed to process order"
    if response.output[0].name not in [tool["name"] for tool in tools]:
        return "Unknown function called"
    return False


async def call_llm(
    request: OrderRequest, retry_count: int = 0
) -> ResponseFunctionToolCall:
    """Handle calling OpenAI and retry if needed. Using Async to take advantage of FastAPI's performance benefits"""

    client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = await client.responses.create(
        model="gpt-4.1-nano",
        temperature=0.0,
        instructions=system_message,
        input=[{"role": "user", "content": request.message}],
        tools=tools,  # type: ignore
        tool_choice="required",
        parallel_tool_calls=False,
    )

    problem = llm_bad_response(response)
    if problem:
        if retry_count < 2:
            print(f"Retrying LLM call {retry_count + 1} of 2. Reason: {problem}")
            return await call_llm(request, retry_count + 1)
        else:
            raise HTTPException(status_code=502, detail=problem)

    return response.output[0]  # type: ignore


@app.post("/process_order", response_model=OrderResponse)
async def process_order(request: OrderRequest) -> OrderResponse:
    response = await call_llm(request)
    function_name = response.name
    arguments = json.loads(response.arguments)

    if function_name == "create_order":
        create_order_tool(**arguments)
    elif function_name == "cancel_order":
        cancel_order_tool(**arguments)

    return OrderResponse(orders=list(all_orders.values()))


@app.get("/orders", response_model=OrderResponse)
async def get_orders() -> OrderResponse:
    return OrderResponse(orders=list(all_orders.values()))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", reload=True)
