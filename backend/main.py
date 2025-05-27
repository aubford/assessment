from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import openai
import os
import json
import itertools
import asyncio
import logging
from typing import Literal, Annotated
from datetime import datetime
from contextlib import asynccontextmanager
from openai.types.responses import Response, ResponseFunctionToolCall

# Configure logging for production observability
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Settings(BaseModel):
    """
    Application settings with environment variable support.
    Follows 12-factor app methodology for configuration management.
    """

    openai_api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    llm_rate_limit: int = Field(
        default_factory=lambda: int(os.getenv("LLM_RATE_LIMIT", "10"))
    )
    cors_origins: list[str] = Field(default=["http://localhost:5173"])

    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance - initialized once at startup
settings = Settings()


# Dependency injection functions for FastAPI
async def get_llm_semaphore(request: Request) -> asyncio.Semaphore:
    return request.app.state.llm_semaphore


async def get_openai_client(request: Request) -> openai.AsyncOpenAI:
    return request.app.state.openai_client


async def get_order_state(request: Request) -> tuple[itertools.count, dict]:
    return request.app.state.order_id_generator, request.app.state.all_orders


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifecycle - startup and shutdown.
    This is the recommended way to initialize resources that need
    to be available throughout the application lifetime.
    """
    logger.info("Starting up application...")

    # Startup: Initialize shared resources
    # Rate limiting semaphore - controls concurrent LLM API calls
    app.state.llm_semaphore = asyncio.Semaphore(settings.llm_rate_limit)
    logger.info(
        f"Initialized LLM rate limiter with {settings.llm_rate_limit} concurrent calls"
    )

    # OpenAI client - reuse connection pool for efficiency
    app.state.openai_client = openai.AsyncOpenAI(api_key=settings.openai_api_key)
    logger.info("Initialized OpenAI client")

    # Application state - centralized data management
    app.state.order_id_generator = itertools.count(1)
    app.state.all_orders = {}
    logger.info("Initialized order management state")

    logger.info("Application startup complete")
    yield

    # Shutdown: cleanup resources if needed
    logger.info("Shutting down application...")
    # Note: asyncio.Semaphore and OpenAI client don't require explicit cleanup
    logger.info("Application shutdown complete")


# Initialize FastAPI app with proper metadata and lifespan
app = FastAPI(lifespan=lifespan)

# Add CORS middleware with configuration from settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
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


def create_order_tool(
    burgers: int = 0,
    fries: int = 0,
    drinks: int = 0,
    order_state: tuple[itertools.count, dict] | None = None,
) -> None:
    if burgers + fries + drinks < 1:
        raise HTTPException(
            status_code=400, detail="Order must contain at least one item"
        )

    if order_state is None:
        raise ValueError("order_state is required")

    order_id_generator, all_orders = order_state
    order_id = next(order_id_generator)

    order = Order(
        id=order_id,
        burgers=burgers,
        fries=fries,
        drinks=drinks,
        timestamp=datetime.now().isoformat(),
    )

    all_orders[order_id] = order
    logger.info(
        f"Created order #{order_id}: {burgers} burgers, {fries} fries, {drinks} drinks"
    )


def cancel_order_tool(
    order_id: int, order_state: tuple[itertools.count, dict] | None = None
) -> None:
    if order_state is None:
        raise ValueError("order_state is required")

    _, all_orders = order_state

    if order_id not in all_orders:
        raise HTTPException(status_code=404, detail=f"Order #{order_id} not found")

    del all_orders[order_id]
    logger.info(f"Cancelled order #{order_id}")


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
    request: OrderRequest,
    semaphore: Annotated[asyncio.Semaphore, Depends(get_llm_semaphore)],
    client: Annotated[openai.AsyncOpenAI, Depends(get_openai_client)],
    retry_count: int = 0,
) -> ResponseFunctionToolCall:
    """
    Handle calling OpenAI with proper rate limiting and retry logic.
    Uses dependency injection for better testability and resource management.
    """
    logger.info(f"Processing LLM request, retry_count={retry_count}")

    async with semaphore:
        try:
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
                    logger.warning(
                        f"Retrying LLM call {retry_count + 1} of 2. Reason: {problem}"
                    )
                    return await call_llm(request, semaphore, client, retry_count + 1)
                else:
                    logger.error(f"LLM call failed after retries: {problem}")
                    raise HTTPException(
                        status_code=502, detail=f"Order processing failed: {problem}"
                    )

            logger.info("LLM request successful")
            return response.output[0]  # type: ignore

        except Exception as e:
            logger.error(f"LLM request failed: {e}")
            if retry_count < 2:
                logger.warning(
                    f"Retrying LLM call {retry_count + 1} of 2 due to exception"
                )
                return await call_llm(request, semaphore, client, retry_count + 1)
            else:
                raise HTTPException(status_code=502, detail=str(e))


@app.post(
    "/process_order",
    response_model=OrderResponse,
    summary="Process a drive-thru order",
    description="Process customer order using LLM to determine create/cancel action",
)
async def process_order(
    request: OrderRequest,
    order_state: Annotated[tuple[itertools.count, dict], Depends(get_order_state)],
    semaphore: Annotated[asyncio.Semaphore, Depends(get_llm_semaphore)],
    client: Annotated[openai.AsyncOpenAI, Depends(get_openai_client)],
) -> OrderResponse:
    """
    Process a customer order request using LLM natural language understanding.
    Demonstrates proper dependency injection and error handling.
    """
    logger.info(f"Processing order request: {request.message[:100]}...")

    try:
        # Call LLM with dependency injection
        response = await call_llm(request, semaphore, client)
        function_name = response.name
        arguments = json.loads(response.arguments)

        # Execute the appropriate tool function
        if function_name == "create_order":
            create_order_tool(order_state=order_state, **arguments)
        elif function_name == "cancel_order":
            cancel_order_tool(order_state=order_state, **arguments)
        else:
            raise HTTPException(
                status_code=400, detail=f"Unknown function: {function_name}"
            )

        # Return current orders
        _, all_orders = order_state
        logger.info(f"Order processed successfully. Total orders: {len(all_orders)}")
        return OrderResponse(orders=list(all_orders.values()))

    except Exception as e:
        logger.error(f"Failed to process order: {e}")
        raise


@app.get(
    "/orders",
    response_model=OrderResponse,
    summary="Get all orders",
    description="Retrieve all current orders in the system",
)
async def get_orders(
    order_state: Annotated[tuple[itertools.count, dict], Depends(get_order_state)],
) -> OrderResponse:
    """
    Get all current orders.
    Demonstrates proper dependency injection for state access.
    """
    _, all_orders = order_state
    logger.info(f"Retrieved {len(all_orders)} orders")
    return OrderResponse(orders=list(all_orders.values()))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", reload=True)
