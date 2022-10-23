import asyncio
import time
from cProfile import run


async def count():
    print("One")
    await asyncio.sleep(1)
    print("Two")


async def main():
    loop = asyncio.get_event_loop()

    t1 = loop.create_task(count())
    t2 = loop.create_task(count())
    t3 = loop.create_task(count())

    await t1
    await t2
    await t3


def run_example():
    s = time.perf_counter()
    asyncio.run(main())
    elapsed = time.perf_counter() - s
    print(f"executed in {elapsed:0.2f} seconds.")


if __name__ == "__main__":
    run_example()
