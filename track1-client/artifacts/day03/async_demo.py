import asyncio


async def fake_inference(
    request_id: int,
    duration: float,
) -> str:
    """Simulate an inference request."""

    print(f"Request {request_id}: started")

    try:
        await asyncio.sleep(duration)
        print(f"Request {request_id}: completed")
        return f"Response from request {request_id}"

    finally:
        print(f"Request {request_id}: cleanup executed")


async def run_demo() -> None:
    """Run concurrent requests with a deadline."""

    tasks = [
        asyncio.create_task(fake_inference(1, 2.0)),
        asyncio.create_task(fake_inference(2, 1.0)),
        asyncio.create_task(fake_inference(3, 5.0)),
    ]

    try:
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks),
                timeout=3.0,
            )
            print("Results:", results)

        except TimeoutError:
            print("Timeout: requests exceeded the deadline")

    finally:
        print("Cleaning up unfinished tasks")

        for task in tasks:
            if not task.done():
                task.cancel()

        await asyncio.gather(*tasks, return_exceptions=True)

        print("Cleanup finished")
async def cancellation_demo() -> None:
    """Demonstrate explicit task cancellation."""

    task = asyncio.create_task(fake_inference(4, 10.0))

    # Give the task time to start.
    await asyncio.sleep(0.1)

    print("Request 4: cancellation requested")

    # Request cancellation.
    task.cancel()

    try:
        await task

    except asyncio.CancelledError:
        print("Request 4: cancellation confirmed")

    print("Request 4 done:", task.done())
    print("Request 4 cancelled:", task.cancelled())

if __name__ == "__main__":
    asyncio.run(cancellation_demo())